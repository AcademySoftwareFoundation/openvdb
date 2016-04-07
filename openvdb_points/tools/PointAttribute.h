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
/// @author Dan Bailey
///
/// @file PointAttribute.h
///
/// @brief  Point attribute manipulation in a VDB Point Grid.
///


#ifndef OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/AttributeGroup.h>
#include <openvdb_points/tools/PointDataGrid.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Appends a new attribute to the VDB tree.
///
/// @param tree          the PointDataTree to be appended to.
/// @param newAttribute  name and type for the new attribute.
/// @param defaultValue  metadata default attribute value
/// @param hidden        mark attribute as hidden
/// @param transient     mark attribute as transient
/// @param group         mark attribute as group
template <typename PointDataTree>
inline void appendAttribute(PointDataTree& tree,
                            const AttributeSet::Util::NameAndType& newAttribute,
                            Metadata::Ptr defaultValue = Metadata::Ptr(),
                            const bool hidden = false,
                            const bool transient = false,
                            const bool group = false);

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param indices       indices of the attributes to drop.
template <typename PointDataTree>
inline void dropAttributes( PointDataTree& tree,
                            const std::vector<size_t>& indices);

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param names         names of the attributes to drop.
template <typename PointDataTree>
inline void dropAttributes( PointDataTree& tree,
                            const std::vector<Name>& names);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param index         index of the attribute to drop.
template <typename PointDataTree>
inline void dropAttribute(  PointDataTree& tree,
                            const size_t& index);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param name          name of the attribute to drop.
template <typename PointDataTree>
inline void dropAttribute(  PointDataTree& tree,
                            const Name& name);

/// @brief Rename attributes in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldNames      a list of old attribute names to rename from.
/// @param newNames      a list of new attribute names to rename to.
///
/// @note Number of oldNames must match the number of newNames.
///
/// @note Duplicate names and renaming group attributes are not allowed.
template <typename PointDataTree>
inline void renameAttributes(PointDataTree& tree,
                            const std::vector<Name>& oldNames,
                            const std::vector<Name>& newNames);

/// @brief Rename an attribute in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldName       the old attribute name to rename from.
/// @param newName       the new attribute name to rename to.
///
/// @note newName must not already exist and must not be a group attribute.
template <typename PointDataTree>
inline void renameAttribute(PointDataTree& tree,
                            const Name& oldName,
                            const Name& newName);

/// @brief Compact attributes in a VDB tree (if possible).
///
/// @param tree          the PointDataTree.
template <typename PointDataTree>
inline void compactAttributes(PointDataTree& tree);

/// @brief Apply Blosc compression to one attribute in the VDB tree.
///
/// @param tree          the PointDataTree.
/// @param name          name of the attribute to compress.
template <typename PointDataTree>
inline void bloscCompressAttribute( PointDataTree& tree,
                                    const Name& name);

////////////////////////////////////////


namespace point_attribute_internal {

template<typename PointDataTreeType>
struct AppendAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;
    typedef AttributeSet::Descriptor::NameAndType               NameAndType;

    AppendAttributeOp(  PointDataTreeType& tree,
                        const NameAndType& newAttribute,
                        AttributeSet::DescriptorPtr& descriptor,
                        const bool hidden = false,
                        const bool transient = false,
                        const bool group = false)
        : mTree(tree)
        , mNewAttribute(newAttribute)
        , mDescriptor(descriptor)
        , mHidden(hidden)
        , mTransient(transient)
        , mGroup(group) { }

    void operator()(const LeafRangeT& range) const {

        for (typename LeafRangeT::Iterator leaf=range.begin(); leaf; ++leaf) {

            const AttributeSet::Descriptor& expected = leaf->attributeSet().descriptor();

            AttributeArray::Ptr attribute = leaf->appendAttribute(mNewAttribute, expected, mDescriptor);

            if (mHidden)      attribute->setHidden(true);
            if (mTransient)   attribute->setTransient(true);

            if (mGroup) {
                GroupAttributeArray::cast(*attribute).setGroup(true);
            }
        }
    }

    //////////

    PointDataTreeType&              mTree;
    const NameAndType&              mNewAttribute;
    AttributeSet::DescriptorPtr&    mDescriptor;
    const bool                      mHidden;
    const bool                      mTransient;
    const bool                      mGroup;
}; // class AppendAttributeOp


////////////////////////////////////////


template<typename PointDataTreeType>
struct DropAttributesOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;
    typedef std::vector<size_t>                                 Indices;

    DropAttributesOp(   PointDataTreeType& tree,
                        const Indices& indices,
                        AttributeSet::DescriptorPtr& descriptor)
        : mTree(tree)
        , mIndices(indices)
        , mDescriptor(descriptor) { }

    void operator()(const LeafRangeT& range) const {

        for (typename LeafRangeT::Iterator leaf=range.begin(); leaf; ++leaf) {

            const AttributeSet::Descriptor& expected = leaf->attributeSet().descriptor();

            leaf->dropAttributes(mIndices, expected, mDescriptor);
        }
    }

    //////////

    PointDataTreeType&              mTree;
    const Indices&                  mIndices;
    AttributeSet::DescriptorPtr&    mDescriptor;
}; // class DropAttributesOp


////////////////////////////////////////


template<typename PointDataTreeType>
struct CompactAttributesOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;

    CompactAttributesOp() { }

    void operator()(const LeafRangeT& range) const {
        for (typename LeafRangeT::Iterator leaf=range.begin(); leaf; ++leaf) {
            leaf->compactAttributes();
        }
    }
}; // class CompactAttributesOp


////////////////////////////////////////


template<typename PointDataTreeType>
struct BloscCompressAttributesOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;
    typedef std::vector<size_t>                                 Indices;

    BloscCompressAttributesOp(  PointDataTreeType& tree,
                                const Indices& indices)
        : mTree(tree)
        , mIndices(indices) { }

    void operator()(const LeafRangeT& range) const {

        for (typename LeafRangeT::Iterator leaf=range.begin(); leaf; ++leaf) {

            for (Indices::const_iterator    it = mIndices.begin(),
                                            itEnd = mIndices.end(); it != itEnd; ++it) {

                AttributeArray& array = leaf->attributeArray(*it);
                array.compress();
            }
        }
    }

    //////////

    PointDataTreeType&              mTree;
    const Indices&                  mIndices;
}; // class BloscCompressAttributesOp


} // namespace point_attribute_internal


////////////////////////////////////////


template <typename PointDataTree>
inline void appendAttribute(PointDataTree& tree,
                            const AttributeSet::Util::NameAndType& newAttribute,
                            Metadata::Ptr defaultValue,
                            const bool hidden, const bool transient, const bool group)
{
    typedef AttributeSet::Util::NameAndTypeVec                    NameAndTypeVec;
    typedef AttributeSet::Descriptor                              Descriptor;

    using point_attribute_internal::AppendAttributeOp;

    typename PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return;

    // do not append a non-unique attribute

    const Descriptor& descriptor = iter->attributeSet().descriptor();
    const size_t index = descriptor.find(newAttribute.name);

    if (index != AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Cannot append an attribute with a non-unique name - " << newAttribute.name << ".");
    }

    // create a new attribute descriptor
    NameAndTypeVec vec;
    vec.push_back(newAttribute);

    Descriptor::Ptr newDescriptor = descriptor.duplicateAppend(vec);

    // store the attribute default value in the descriptor metadata

    if (defaultValue) {
        newDescriptor->setDefaultValue(newAttribute.name, *defaultValue);
    }

    // insert attributes using the new descriptor

    AppendAttributeOp<PointDataTree> append(tree, newAttribute, newDescriptor, hidden, transient, group);
    tbb::parallel_for(typename tree::template LeafManager<PointDataTree>(tree).leafRange(), append);
}


////////////////////////////////////////


template <typename PointDataTree>
inline void dropAttributes( PointDataTree& tree,
                            const std::vector<size_t>& indices)
{
    typedef typename tree::LeafManager<PointDataTree>       LeafManagerT;
    typedef AttributeSet::Descriptor                        Descriptor;

    using point_attribute_internal::DropAttributesOp;

    typename PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return;

    const Descriptor& descriptor = iter->attributeSet().descriptor();

    // throw if position index present in the indices as this attribute is mandatory

    const size_t positionIndex = descriptor.find("P");
    if (positionIndex!= AttributeSet::INVALID_POS &&
        std::find(indices.begin(), indices.end(), positionIndex) != indices.end()) {
        OPENVDB_THROW(KeyError, "Cannot drop mandatory position attribute.");
    }

    // insert attributes using the new descriptor

    Descriptor::Ptr newDescriptor = descriptor.duplicateDrop(indices);
    tbb::parallel_for(LeafManagerT(tree).leafRange(), DropAttributesOp<PointDataTree>(tree, indices, newDescriptor));
}


////////////////////////////////////////


template <typename PointDataTree>
inline void dropAttributes( PointDataTree& tree,
                            const std::vector<Name>& names)
{
    typename PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();

    std::vector<size_t> indices;

    for (std::vector<Name>::const_iterator it = names.begin(), itEnd = names.end(); it != itEnd; ++it) {
        const size_t index = descriptor.find(*it);

        // do not attempt to drop an attribute that does not exist
        if (index == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError, "Cannot drop an attribute that does not exist - " << *it << ".");
        }

        indices.push_back(index);
    }

    dropAttributes(tree, indices);
}


////////////////////////////////////////


template <typename PointDataTree>
inline void dropAttribute(  PointDataTree& tree,
                            const size_t& index)
{
    std::vector<size_t> indices;
    indices.push_back(index);
    dropAttributes(tree, indices);
}


template <typename PointDataTree>
inline void dropAttribute(  PointDataTree& tree,
                            const Name& name)
{
    std::vector<Name> names;
    names.push_back(name);
    dropAttributes(tree, names);
}


////////////////////////////////////////


template <typename PointDataTree>
inline void renameAttributes(   PointDataTree& tree,
                                const std::vector<Name>& oldNames,
                                const std::vector<Name>& newNames)
{
    if (oldNames.size() != newNames.size()) {
        OPENVDB_THROW(ValueError, "Mis-matching sizes of name vectors, cannot rename attributes.");
    }

    typedef AttributeSet::Descriptor                        Descriptor;

    typename PointDataTree::LeafIter iter = tree.beginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor& descriptor = attributeSet.descriptor();
    AttributeSet::DescriptorPtr newDescriptor(new Descriptor(descriptor));

    for (size_t i = 0; i < oldNames.size(); i++) {
        const Name oldName(oldNames[i]);
        if (descriptor.find(oldName) == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError, "Cannot find requested attribute - " << oldName << ".");
        }

        const Name newName(newNames[i]);
        if (descriptor.find(newName) != AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError, "Cannot rename attribute as new name already exists - " << newName << ".");
        }

        const AttributeArray* array = attributeSet.getConst(oldName);
        assert(array);

        if (GroupAttributeArray::isGroup(*array)) {
            OPENVDB_THROW(KeyError, "Cannot rename group attribute - " << oldName << ".");
        }

        newDescriptor->rename(oldName, newName);
    }

    for (; iter; ++iter) {
        iter->renameAttributes(descriptor, newDescriptor);
    }
}


template <typename PointDataTree>
inline void renameAttribute(PointDataTree& tree,
                            const Name& oldName,
                            const Name& newName)
{
    std::vector<Name> oldNames;
    std::vector<Name> newNames;
    oldNames.push_back(oldName);
    newNames.push_back(newName);
    renameAttributes(tree, oldNames, newNames);
}


////////////////////////////////////////


template <typename PointDataTree>
inline void compactAttributes(PointDataTree& tree)
{
    typedef typename tree::LeafManager<PointDataTree>       LeafManagerT;

    using point_attribute_internal::CompactAttributesOp;

    typename PointDataTree::LeafIter iter = tree.beginLeaf();
    if (!iter)  return;

    tbb::parallel_for(LeafManagerT(tree).leafRange(), CompactAttributesOp<PointDataTree>());
}


////////////////////////////////////////


template <typename PointDataTree>
inline void bloscCompressAttribute( PointDataTree& tree,
                                    const Name& name)
{
    using point_attribute_internal::BloscCompressAttributesOp;

    typedef typename tree::LeafManager<PointDataTree>       LeafManagerT;
    typedef AttributeSet::Descriptor                        Descriptor;

    typename PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return;

    const Descriptor& descriptor = iter->attributeSet().descriptor();

    // throw if index cannot be found in descriptor

    const size_t index = descriptor.find(name);
    if (index == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Cannot find requested attribute - " << name << ".");
    }

    // blosc compress attributes

    std::vector<size_t> indices;
    indices.push_back(index);

    tbb::parallel_for(LeafManagerT(tree).leafRange(), BloscCompressAttributesOp<PointDataTree>(tree, indices));
}

////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
