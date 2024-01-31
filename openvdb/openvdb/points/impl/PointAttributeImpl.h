// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey, Khang Ngo
///
/// @file PointAttributeImpl.h
///

#ifndef OPENVDB_POINTS_POINT_ATTRIBUTE_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_ATTRIBUTE_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_attribute_internal {


template <typename ValueType>
inline void collapseAttribute(AttributeArray& array,
    const AttributeSet::Descriptor&, const ValueType& uniformValue)
{
    AttributeWriteHandle<ValueType> handle(array);
    handle.collapse(uniformValue);
}


inline void collapseAttribute(AttributeArray& array,
    const AttributeSet::Descriptor& descriptor, const Name& uniformValue)
{
    StringAttributeWriteHandle handle(array, descriptor.getMetadata());
    handle.collapse(uniformValue);
}


////////////////////////////////////////


template <typename ValueType, typename CodecType>
struct AttributeTypeConversion
{
    static const NamePair& type() {
        return TypedAttributeArray<ValueType, CodecType>::attributeType();
    }
};


template <typename CodecType>
struct AttributeTypeConversion<Name, CodecType>
{
    static const NamePair& type() { return StringAttributeArray::attributeType(); }
};


////////////////////////////////////////


template <typename PointDataTreeT, typename ValueType>
struct MetadataStorage
{
    static void add(PointDataTreeT&, const ValueType&) {}

    template<typename AttributeListType>
    static void add(PointDataTreeT&, const AttributeListType&) {}
};


template <typename PointDataTreeT>
struct MetadataStorage<PointDataTreeT, Name>
{
    static void add(PointDataTreeT& tree, const Name& uniformValue) {
        MetaMap& metadata = makeDescriptorUnique(tree)->getMetadata();
        StringMetaInserter inserter(metadata);
        inserter.insert(uniformValue);
    }

    template<typename AttributeListType>
    static void add(PointDataTreeT& tree, const AttributeListType& data) {
        MetaMap& metadata = makeDescriptorUnique(tree)->getMetadata();
        StringMetaInserter inserter(metadata);
        Name value;

        for (size_t i = 0; i < data.size(); i++) {
            data.get(value, i);
            inserter.insert(value);
        }
    }
};


} // namespace point_attribute_internal

/// @endcond


////////////////////////////////////////


template <typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const Name& name,
                            const NamePair& type,
                            const Index strideOrTotalSize,
                            const bool constantStride,
                            const Metadata* defaultValue,
                            const bool hidden,
                            const bool transient)
{
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    // do not append a non-unique attribute

    const auto& descriptor = iter->attributeSet().descriptor();
    const size_t index = descriptor.find(name);

    if (index != AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError,
            "Cannot append an attribute with a non-unique name - " << name << ".");
    }

    // create a new attribute descriptor

    auto newDescriptor = descriptor.duplicateAppend(name, type);

    // store the attribute default value in the descriptor metadata

    if (defaultValue) {
        newDescriptor->setDefaultValue(name, *defaultValue);
    }

    // extract new pos

    const size_t pos = newDescriptor->find(name);

    // acquire registry lock to avoid locking when appending attributes in parallel

    AttributeArray::ScopedRegistryLock lock;

    // insert attributes using the new descriptor

    tree::LeafManager<PointDataTreeT> leafManager(tree);
    leafManager.foreach(
        [&](typename PointDataTreeT::LeafNodeType& leaf, size_t /*idx*/) {
            auto expected = leaf.attributeSet().descriptorPtr();

            auto attribute = leaf.appendAttribute(*expected, newDescriptor,
                pos, strideOrTotalSize, constantStride, defaultValue,
                &lock);

            if (hidden)     attribute->setHidden(true);
            if (transient)  attribute->setTransient(true);
        }, /*threaded=*/ true
    );
}


////////////////////////////////////////


template <typename ValueType, typename CodecType, typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const std::string& name,
                            const ValueType& uniformValue,
                            const Index strideOrTotalSize,
                            const bool constantStride,
                            const TypedMetadata<ValueType>* defaultValue,
                            const bool hidden,
                            const bool transient)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");

    using point_attribute_internal::AttributeTypeConversion;
    using point_attribute_internal::Default;
    using point_attribute_internal::MetadataStorage;

    appendAttribute(tree, name, AttributeTypeConversion<ValueType, CodecType>::type(),
        strideOrTotalSize, constantStride, defaultValue, hidden, transient);

    // if the uniform value is equal to either the default value provided
    // through the metadata argument or the default value for this value type,
    // it is not necessary to perform the collapse

    const bool uniformIsDefault = math::isExactlyEqual(uniformValue,
            bool(defaultValue) ? defaultValue->value() : Default<ValueType>::value());
    if (!uniformIsDefault) {
        MetadataStorage<PointDataTreeT, ValueType>::add(tree, uniformValue);
        collapseAttribute<ValueType>(tree, name, uniformValue);
    }
}


////////////////////////////////////////


template <typename ValueType, typename PointDataTreeT>
inline void collapseAttribute(  PointDataTreeT& tree,
                                const Name& name,
                                const ValueType& uniformValue)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const auto& descriptor = iter->attributeSet().descriptor();

    // throw if attribute name does not exist

    const size_t index = descriptor.find(name);
    if (index == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Cannot find attribute name in PointDataTree.");
    }

    tree::LeafManager<PointDataTreeT> leafManager(tree);
    leafManager.foreach(
        [&](typename PointDataTreeT::LeafNodeType& leaf, size_t /*idx*/) {
            OPENVDB_ASSERT(leaf.hasAttribute(index));
            AttributeArray& array = leaf.attributeArray(index);
            point_attribute_internal::collapseAttribute(
                array, descriptor, uniformValue);
        }, /*threaded=*/true
    );
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<size_t>& indices)
{
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const auto& descriptor = iter->attributeSet().descriptor();

    // throw if position index present in the indices as this attribute is mandatory

    const size_t positionIndex = descriptor.find("P");
    if (positionIndex!= AttributeSet::INVALID_POS &&
        std::find(indices.begin(), indices.end(), positionIndex) != indices.end()) {
        OPENVDB_THROW(KeyError, "Cannot drop mandatory position attribute.");
    }

    // insert attributes using the new descriptor

    auto newDescriptor = descriptor.duplicateDrop(indices);

    tree::LeafManager<PointDataTreeT> leafManager(tree);
    leafManager.foreach(
        [&](typename PointDataTreeT::LeafNodeType& leaf, size_t /*idx*/) {
            auto expected = leaf.attributeSet().descriptorPtr();
            leaf.dropAttributes(indices, *expected, newDescriptor);
        }, /*threaded=*/true
    );
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<Name>& names)
{
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();

    std::vector<size_t> indices;

    for (const Name& name : names) {
        const size_t index = descriptor.find(name);

        // do not attempt to drop an attribute that does not exist
        if (index == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError,
                "Cannot drop an attribute that does not exist - " << name << ".");
        }

        indices.push_back(index);
    }

    dropAttributes(tree, indices);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const size_t& index)
{
    std::vector<size_t> indices{index};
    dropAttributes(tree, indices);
}


template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const Name& name)
{
    std::vector<Name> names{name};
    dropAttributes(tree, names);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void renameAttributes(   PointDataTreeT& tree,
                                const std::vector<Name>& oldNames,
                                const std::vector<Name>& newNames)
{
    if (oldNames.size() != newNames.size()) {
        OPENVDB_THROW(ValueError, "Mis-matching sizes of name vectors, cannot rename attributes.");
    }

    using Descriptor = AttributeSet::Descriptor;

    auto iter = tree.beginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor::Ptr descriptor = attributeSet.descriptorPtr();
    auto newDescriptor = std::make_shared<Descriptor>(*descriptor);

    for (size_t i = 0; i < oldNames.size(); i++) {
        const Name& oldName = oldNames[i];
        if (descriptor->find(oldName) == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError, "Cannot find requested attribute - " << oldName << ".");
        }

        const Name& newName = newNames[i];
        if (descriptor->find(newName) != AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError,
                "Cannot rename attribute as new name already exists - " << newName << ".");
        }

        const AttributeArray* array = attributeSet.getConst(oldName);
        OPENVDB_ASSERT(array);

        if (isGroup(*array)) {
            OPENVDB_THROW(KeyError, "Cannot rename group attribute - " << oldName << ".");
        }

        newDescriptor->rename(oldName, newName);
    }

    for (; iter; ++iter) {
        iter->renameAttributes(*descriptor, newDescriptor);
    }
}


template <typename PointDataTreeT>
inline void renameAttribute(PointDataTreeT& tree,
                            const Name& oldName,
                            const Name& newName)
{
    renameAttributes(tree, {oldName}, {newName});
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void compactAttributes(PointDataTreeT& tree)
{
    auto iter = tree.beginLeaf();
    if (!iter)  return;

    tree::LeafManager<PointDataTreeT> leafManager(tree);
    leafManager.foreach(
        [&](typename PointDataTreeT::LeafNodeType& leaf, size_t /*idx*/) {
            leaf.compactAttributes();
        }, /*threaded=*/ true
    );
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_ATTRIBUTE_IMPL_HAS_BEEN_INCLUDED
