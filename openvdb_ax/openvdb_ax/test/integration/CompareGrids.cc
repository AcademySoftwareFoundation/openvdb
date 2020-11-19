// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file test/integration/CompareGrids.cc

#include "CompareGrids.h"

#include <openvdb/points/PointDataGrid.h>

namespace unittest_util
{

#if (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER == 7 && \
     OPENVDB_LIBRARY_MINOR_VERSION_NUMBER == 1)
// Issue with TypeList where Unqiue defines recursively in 7.1
template <typename... Ts> struct TListFix;

template<typename ListT, typename... Ts> struct TSAppendImpl;
template<typename... Ts, typename... OtherTs>
struct TSAppendImpl<TListFix<Ts...>, OtherTs...> { using type = TListFix<Ts..., OtherTs...>; };
template<typename... Ts, typename... OtherTs>
struct TSAppendImpl<TListFix<Ts...>, TListFix<OtherTs...>> { using type = TListFix<Ts..., OtherTs...>; };

template<typename ListT, typename T> struct TSEraseImpl;
template<typename T> struct TSEraseImpl<TListFix<>, T> { using type = TListFix<>; };
template<typename... Ts, typename T>
struct TSEraseImpl<TListFix<T, Ts...>, T> { using type = typename TSEraseImpl<TListFix<Ts...>, T>::type; };
template<typename T2, typename... Ts, typename T>
struct TSEraseImpl<TListFix<T2, Ts...>, T> {
    using type = typename TSAppendImpl<TListFix<T2>,
        typename TSEraseImpl<TListFix<Ts...>, T>::type>::type;
};

template<typename ListT, typename... Ts> struct TSRemoveImpl;
template<typename ListT> struct TSRemoveImpl<ListT> { using type = ListT; };
template<typename ListT, typename T, typename... Ts>
struct TSRemoveImpl<ListT, T, Ts...> { using type = typename TSRemoveImpl<typename TSEraseImpl<ListT, T>::type, Ts...>::type; };
template<typename ListT, typename... Ts>
struct TSRemoveImpl<ListT, TListFix<Ts...>> { using type = typename TSRemoveImpl<ListT, Ts...>::type; };

template <typename... Ts>
struct TListFix
{
    using Self = TListFix;
    template<typename... TypesToRemove>
    using Remove = typename TSRemoveImpl<Self, TypesToRemove...>::type;
    template<typename... TypesToAppend>
    using Append = typename TSAppendImpl<Self, TypesToAppend...>::type;
    template<typename OpT>
    static void foreach(OpT op) { openvdb::internal::TSForEachImpl<OpT, Ts...>(op); }
};
using TypeList = TListFix<
#else
using TypeList = openvdb::TypeList<
#endif
    double,
    float,
    int64_t,
    int32_t,
    int16_t,
    bool,
    openvdb::math::Vec2<double>,
    openvdb::math::Vec2<float>,
    openvdb::math::Vec2<int32_t>,
    openvdb::math::Vec3<double>,
    openvdb::math::Vec3<float>,
    openvdb::math::Vec3<int32_t>,
    openvdb::math::Vec4<double>,
    openvdb::math::Vec4<float>,
    openvdb::math::Vec4<int32_t>,
    openvdb::math::Mat3<double>,
    openvdb::math::Mat3<float>,
    openvdb::math::Mat4<double>,
    openvdb::math::Mat4<float>,
    std::string>;

struct DiagnosticArrayData
{
    DiagnosticArrayData()
        : mSizeMatch(true)
        , mTypesMatch(true)
        , mFlagsMatch(true)
        , mArrayValueFlags() {}

    inline void
    flagArrayValue(const size_t idx) {
        if (!mArrayValueFlags) mArrayValueFlags.reset(new std::vector<size_t>());
        (*mArrayValueFlags).push_back(idx);
    }

    bool mSizeMatch;
    bool mTypesMatch;
    bool mFlagsMatch;
    std::unique_ptr<std::vector<size_t>> mArrayValueFlags;
};

struct DiagnosticData
{
    using Ptr = std::shared_ptr<DiagnosticData>;

    DiagnosticData()
        : mValid(true)
        , mBufferSizes(true)
        , mVoxelTopologyFlags(nullptr)
        , mVoxelValueFlags(nullptr)
        , mDescriptorsMatch(true)
        , mAttributeArrayData() {}

    inline bool hasValueFlags() const {
        return static_cast<bool>(mVoxelValueFlags);
    }

    inline bool hasTopologyFlags() const {
        return static_cast<bool>(mVoxelTopologyFlags);
    }

    inline void
    flagVoxelTopology(const int16_t idx) {
        if (!mVoxelTopologyFlags) {
            mVoxelTopologyFlags.reset(new std::array<bool,512>());
            mVoxelTopologyFlags->fill(true);
        }
        (*mVoxelTopologyFlags)[idx] = false;
    }

    inline void
    flagVoxelValue(const int16_t idx) {
        if (!mVoxelValueFlags) {
            mVoxelValueFlags.reset(new std::array<bool,512>());
            mVoxelValueFlags->fill(true);
        }
        (*mVoxelValueFlags)[idx] = false;
    }

    inline DiagnosticArrayData&
    getDiagnosticArrayData(const std::string& name) {
        if (!mAttributeArrayData) {
            mAttributeArrayData.reset(new std::map<std::string, DiagnosticArrayData>());
        }
        return (*mAttributeArrayData)[name];
    }

    inline bool
    hasDiagnosticArrayData() const {
        return (static_cast<bool>(mAttributeArrayData));
    }

    inline bool
    hasDiagnosticArrayData(const std::string& name) const {
        return (hasDiagnosticArrayData() &&
            mAttributeArrayData->find(name) != mAttributeArrayData->end());
    }

    bool mValid;
    bool mBufferSizes;
    std::unique_ptr<std::array<bool,512>> mVoxelTopologyFlags;
    std::unique_ptr<std::array<bool,512>> mVoxelValueFlags;

    bool mDescriptorsMatch;
    std::unique_ptr<std::map<std::string, DiagnosticArrayData>> mAttributeArrayData;
};

template <typename LeafNodeType,
          typename NodeMaskT>
inline bool compareLeafBuffers(const LeafNodeType& firstLeaf,
                               const LeafNodeType& secondLeaf,
                               const NodeMaskT& mask,
                               DiagnosticData& data,
                               const ComparisonSettings& settings,
                               const typename LeafNodeType::ValueType& tolerance)
{
    using BufferT = typename LeafNodeType::Buffer;

    const BufferT& firstBuffer = firstLeaf.buffer();
    const BufferT& secondBuffer = secondLeaf.buffer();

    // if the buffers are not the same size the buffer most likely isn't
    // loaded or allocated

    if (firstBuffer.size() != secondBuffer.size()) {
        data.mBufferSizes = false;
        return false;
    }

    const NodeMaskT& firstMask = firstLeaf.getValueMask();
    const NodeMaskT& secondMask = secondLeaf.getValueMask();
    typename NodeMaskT::OnIterator iter = mask.beginOn();

    for (; iter; ++iter) {
        const openvdb::Index n = iter.pos();
        assert(n < firstBuffer.size() && n < secondBuffer.size());

        if (settings.mCheckActiveStates &&
            firstMask.isOn(n) ^ secondMask.isOn(n)) {
            data.flagVoxelTopology(static_cast<int16_t>(n));
        }

        if (settings.mCheckBufferValues &&
            !openvdb::math::isApproxEqual(firstBuffer[n], secondBuffer[n], tolerance)) {
            data.flagVoxelValue(static_cast<int16_t>(n));
        }
    }

    return !data.hasValueFlags() && !data.hasTopologyFlags();
}

void compareStringArrays(const openvdb::points::AttributeArray& a1,
                         const openvdb::points::AttributeArray& a2,
                         const openvdb::points::PointDataTree::LeafNodeType& leaf1,
                         const openvdb::points::PointDataTree::LeafNodeType& leaf2,
                         const std::string& name,
                         DiagnosticData& data)
{
    using LeafNodeT = openvdb::points::PointDataTree::LeafNodeType;

    if (a1.size() != a2.size()) {
        auto& arrayData = data.getDiagnosticArrayData(name);
        arrayData.mSizeMatch = false;
    }

    const openvdb::points::AttributeSet::Descriptor& descriptor1 = leaf1.attributeSet().descriptor();
    const openvdb::points::AttributeSet::Descriptor& descriptor2 = leaf2.attributeSet().descriptor();

    openvdb::points::StringAttributeHandle h1(a1, descriptor1.getMetadata()), h2(a2, descriptor2.getMetadata());
    auto iter = leaf1.beginIndexAll();

    for (; iter; ++iter) {
        if (h1.get(*iter) != h2.get(*iter)) break;
    }

    if (iter) {
        auto& arrayData = data.getDiagnosticArrayData(name);
        for (; iter; ++iter) {
            const openvdb::Index i = *iter;
            if (h1.get(i) != h2.get(i)) {
                arrayData.flagArrayValue(i);
                data.flagVoxelValue(static_cast<int16_t>(LeafNodeT::coordToOffset(iter.getCoord())));
            }
        }
    }
}

template <typename ValueType>
inline void compareArrays(const openvdb::points::AttributeArray& a1,
                          const openvdb::points::AttributeArray& a2,
                          const openvdb::points::PointDataTree::LeafNodeType& leaf,
                          const std::string& name,
                          DiagnosticData& data)
{
    using LeafNodeT = openvdb::points::PointDataTree::LeafNodeType;

    if (a1.size() != a2.size()) {
        auto& arrayData = data.getDiagnosticArrayData(name);
        arrayData.mSizeMatch = false;
    }

    openvdb::points::AttributeHandle<ValueType> h1(a1), h2(a2);
    auto iter = leaf.beginIndexAll();

    for (; iter; ++iter) {
        if (h1.get(*iter) != h2.get(*iter)) break;
    }

    if (iter) {
        auto& arrayData = data.getDiagnosticArrayData(name);
        for (; iter; ++iter) {
            const openvdb::Index i = *iter;
            if (h1.get(i) != h2.get(i)) {
                arrayData.flagArrayValue(i);
                data.flagVoxelValue(static_cast<int16_t>(LeafNodeT::coordToOffset(iter.getCoord())));
            }
        }
    }
}

template <typename LeafNodeType>
inline bool
compareAttributes(const LeafNodeType&,
                  const LeafNodeType&,
                  DiagnosticData&,
                  const ComparisonSettings&) {
    return true;
}

template <>
inline bool
compareAttributes<openvdb::points::PointDataTree::LeafNodeType>
    (const openvdb::points::PointDataTree::LeafNodeType& firstLeaf,
     const openvdb::points::PointDataTree::LeafNodeType& secondLeaf,
     DiagnosticData& data,
     const ComparisonSettings& settings)
{
    using Descriptor = openvdb::points::AttributeSet::Descriptor;

    const Descriptor& firstDescriptor = firstLeaf.attributeSet().descriptor();
    const Descriptor& secondDescriptor = secondLeaf.attributeSet().descriptor();

    if (settings.mCheckDescriptors &&
        !firstDescriptor.hasSameAttributes(secondDescriptor)) {
        data.mDescriptorsMatch = false;
    }

    // check common/miss-matching attributes

    std::set<std::string> attrs1, attrs2;
    for (const auto& nameToPos : firstDescriptor.map()) {
        attrs1.insert(nameToPos.first);
    }
    for (const auto& nameToPos : secondDescriptor.map()) {
        attrs2.insert(nameToPos.first);
    }

    std::vector<std::string> commonAttributes;
    std::set_intersection(attrs1.begin(),
        attrs1.end(),
        attrs2.begin(),
        attrs2.end(),
        std::back_inserter(commonAttributes));

    for (const std::string& name : commonAttributes) {
        const size_t pos1 = firstDescriptor.find(name);
        const size_t pos2 = secondDescriptor.find(name);
        const auto& array1 = firstLeaf.constAttributeArray(pos1);
        const auto& array2 = secondLeaf.constAttributeArray(pos2);

        const std::string& type = array1.type().first;
        if (type != array2.type().first) {
            // this mismatch is also loged by differing descriptors
            auto& arrayData = data.getDiagnosticArrayData(name);
            arrayData.mTypesMatch = false;
            continue;
        }

        if (settings.mCheckArrayFlags &&
            array1.flags() != array2.flags()) {
            auto& arrayData = data.getDiagnosticArrayData(name);
            arrayData.mFlagsMatch = false;
        }

        if (settings.mCheckArrayValues) {
            if (array1.type().second == "str") {
                compareStringArrays(array1, array2, firstLeaf, secondLeaf, name, data);
            }
            else {
                bool success = false;
                // Remove string types but add uint8_t types (used by group arrays)
                TypeList::Remove<std::string>::Append<uint8_t>::foreach([&](auto x) {
                    if (type == openvdb::typeNameAsString<decltype(x)>()) {
                        compareArrays<decltype(x)>(array1, array2, firstLeaf, name, data);
                        success = true;
                    }
                });

                if (!success) {
                    throw std::runtime_error("Unsupported array type for comparison: " + type);
                }
            }
        }
    }

    return !data.hasDiagnosticArrayData() && data.mDescriptorsMatch;
}

template<typename TreeType>
struct CompareLeafNodes
{
    using LeafManagerT = openvdb::tree::LeafManager<const openvdb::MaskTree>;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using LeafManagerNodeType = typename LeafManagerT::LeafNodeType;
    using ConstGridAccessor = openvdb::tree::ValueAccessor<const TreeType>;

    CompareLeafNodes(std::vector<DiagnosticData::Ptr>& data,
                     const TreeType& firstTree,
                     const TreeType& secondTree,
                     const typename TreeType::ValueType tolerance,
                     const ComparisonSettings& settings,
                     const bool useVoxelMask = true)
        : mDiagnosticData(data)
        , mFirst(firstTree)
        , mSecond(secondTree)
        , mTolerance(tolerance)
        , mSettings(settings)
        , mUseVoxelMask(useVoxelMask) {}

   void operator()(LeafManagerNodeType& leaf, size_t index) const
   {
       const openvdb::Coord& origin = leaf.origin();
       // // //

       const LeafNodeType* const firstLeafNode = mFirst.probeConstLeaf(origin);
       const LeafNodeType* const secondLeafNode = mSecond.probeConstLeaf(origin);

       if (firstLeafNode == nullptr &&
           secondLeafNode == nullptr) {
           return;
       }

       auto& data = mDiagnosticData[index];
       data.reset(new DiagnosticData());

       if (static_cast<bool>(firstLeafNode) ^
           static_cast<bool>(secondLeafNode)) {
           data->mValid = false;
           return;
       }

       assert(firstLeafNode && secondLeafNode);
       const openvdb::util::NodeMask<LeafNodeType::LOG2DIM>
           mask(mUseVoxelMask ? leaf.valueMask() : true);

       if (compareLeafBuffers(*firstLeafNode, *secondLeafNode, mask, *data, mSettings, mTolerance) &&
           compareAttributes(*firstLeafNode, *secondLeafNode, *data, mSettings)) {
           data.reset();
       }
   }


private:
    std::vector<DiagnosticData::Ptr>&   mDiagnosticData;
    const ConstGridAccessor             mFirst;
    const ConstGridAccessor             mSecond;
    const typename TreeType::ValueType  mTolerance;
    const ComparisonSettings&           mSettings;
    const bool                          mUseVoxelMask;
};

template <typename GridType>
bool compareGrids(ComparisonResult& resultData,
             const GridType& firstGrid,
             const GridType& secondGrid,
             const ComparisonSettings& settings,
             const openvdb::MaskGrid::ConstPtr maskGrid,
             const typename GridType::ValueType tolerance)
{
    using TreeType = typename GridType::TreeType;
    using LeafManagerT = openvdb::tree::LeafManager<const openvdb::MaskTree>;

    struct Local {
        // flag to string
        static std::string fts(const bool flag) {
            return (flag ? "[SUCCESS]" : "[FAILED]");
        }
    };

    bool result = true;
    bool flag = true;
    std::ostream& os = resultData.mOs;

    os << "[Diagnostic : Compare Leaf Nodes Result]"
       << std::endl
       << "    First Grid: \"" << firstGrid.getName() << "\""
       << std::endl
       << "    Second Grid: \"" << secondGrid.getName() << "\""
       << std::endl << std::endl;

    if (firstGrid.tree().hasActiveTiles() ||
        secondGrid.tree().hasActiveTiles()) {
       os << "[Diagnostic : WARNING]: Grids contain active tiles which will not be compared."
          << std::endl;
    }

    if (settings.mCheckTransforms) {
        flag = (firstGrid.constTransform() == secondGrid.constTransform());
        result &= flag;
        os << "[Diagnostic]: Grid transformations: " << Local::fts(flag)
          << std::endl;
    }

    const openvdb::Index64 leafCount1 = firstGrid.tree().leafCount();
    const openvdb::Index64 leafCount2 = secondGrid.tree().leafCount();
    flag = (leafCount1 == 0 && leafCount2 == 0);
    if (flag) {
        os << "[Diagnostic]: Both grids contain 0 leaf nodes."
           << std::endl;
        return result;
    }

    if (settings.mCheckTopologyStructure && !maskGrid) {
        flag = firstGrid.tree().hasSameTopology(secondGrid.tree());
        result &= flag;
         os << "[Diagnostic]: Topology structures: " << Local::fts(flag)
           << std::endl;
    }

    openvdb::MaskGrid::Ptr mask = openvdb::MaskGrid::create();
    if (maskGrid) {
        mask->topologyUnion(*maskGrid);
    }
    else {
        mask->topologyUnion(firstGrid);
        mask->topologyUnion(secondGrid);
    }

    openvdb::tools::pruneInactive(mask->tree());
    LeafManagerT leafManager(mask->constTree());
    std::vector<DiagnosticData::Ptr> data(leafManager.leafCount());

    CompareLeafNodes<TreeType>
        op(data,
           firstGrid.constTree(),
           secondGrid.constTree(),
           tolerance,
           settings);

    leafManager.foreach(op);

    flag = true;
    for (const auto& diagnostic : data) {
        if (diagnostic) {
            flag = false;
            break;
        }
    }

    result &= flag;
    os << "[Diagnostic]: Leaf Node Comparison: " << Local::fts(flag)
       << std::endl;

    if (flag) return result;

    openvdb::MaskGrid& differingTopology = *(resultData.mDifferingTopology);
    openvdb::MaskGrid& differingValues = *(resultData.mDifferingValues);

    differingTopology.setTransform(firstGrid.transform().copy());
    differingValues.setTransform(firstGrid.transform().copy());
    differingTopology.setName("different_topology");
    differingValues.setName("different_values");

    // Print diagnostic info to the stream and intialise the result topologies

    openvdb::MaskGrid::Accessor accessorTopology = differingTopology.getAccessor();
    openvdb::MaskGrid::Accessor accessorValues = differingValues.getAccessor();
    auto range = leafManager.leafRange();

    os << "[Diagnostic]: Leaf Node Diagnostics:"
       << std::endl << std::endl;

    for (auto leaf = range.begin(); leaf; ++leaf) {
        DiagnosticData::Ptr diagnostic = data[leaf.pos()];
        if (!diagnostic) continue;

        const openvdb::Coord& origin = leaf->origin();

        os << "    Coord        : " << origin << std::endl;
        os << "    Both Valid   : " <<  Local::fts(diagnostic->mValid) << std::endl;
        if (!diagnostic->mValid) {
            const bool second = firstGrid.constTree().probeConstLeaf(origin);
            os << "        Missing in " << (second ? "second" : "first")
               << " grid."
               << std::endl;
            continue;
        }

        const auto& l1 = firstGrid.constTree().probeConstLeaf(origin);
        const auto& l2 = secondGrid.constTree().probeConstLeaf(origin);
        assert(l1 && l2);

        os << "    Buffer Sizes : " <<  Local::fts(diagnostic->mBufferSizes) << std::endl;

        const bool topologyMatch = !static_cast<bool>(diagnostic->mVoxelTopologyFlags);
        os << "    Topology     : " <<  Local::fts(topologyMatch) << std::endl;

        if (!topologyMatch) {
            os << "        The following voxel topologies differ : " << std::endl;
            openvdb::Index idx(0);
            for (const auto match : *(diagnostic->mVoxelTopologyFlags)) {
                if (!match) {
                    const openvdb::Coord coord = leaf->offsetToGlobalCoord(idx);
                    os << "            [" << idx << "] "<< coord
                       << " G1: " << l1->isValueOn(coord)
                       << " - G2: " << l2->isValueOn(coord)
                       << std::endl;
                    accessorTopology.setValue(coord, true);
                }
                ++idx;
            }
        }

        const bool valueMatch = !static_cast<bool>(diagnostic->mVoxelValueFlags);
        os << "    Values       : " <<  Local::fts(valueMatch) << std::endl;

        if (!valueMatch) {
            os << "        The following voxel values differ : " << std::endl;
            openvdb::Index idx(0);
            for (const auto match : *(diagnostic->mVoxelValueFlags)) {
                if (!match) {
                    const openvdb::Coord coord = leaf->offsetToGlobalCoord(idx);
                    os << "            [" << idx << "] "<< coord
                       << " G1: " << l1->getValue(coord)
                       << " - G2: " << l2->getValue(coord)
                       << std::endl;
                    accessorValues.setValue(coord, true);
                }
                ++idx;
            }
        }

        if (firstGrid.template isType<openvdb::points::PointDataGrid>()) {
            os << "    Descriptors  : " <<  Local::fts(diagnostic->mDescriptorsMatch) << std::endl;
            const bool attributesMatch = !static_cast<bool>(diagnostic->mAttributeArrayData);
            os << "    Array Data   : " <<  Local::fts(attributesMatch) << std::endl;
            if (!attributesMatch) {
                os << "        The following attribute values : " << std::endl;
                for (const auto& iter : *(diagnostic->mAttributeArrayData)) {

                    const std::string& name = iter.first;
                    const DiagnosticArrayData& arrayData = iter.second;

                    os << "            Attribute Array : [" << name << "] " << std::endl
                       << "            Size Match      : " << Local::fts(arrayData.mSizeMatch) << std::endl
                       << "            Type Match      : " << Local::fts(arrayData.mTypesMatch) << std::endl
                       << "            Flags Match     : " << Local::fts(arrayData.mFlagsMatch) << std::endl;

                    const bool arrayValuesMatch = !static_cast<bool>(arrayData.mArrayValueFlags);
                    os << "            Array Values    : " <<  Local::fts(arrayValuesMatch) << std::endl;
                    if (!arrayValuesMatch) {
                        for (size_t idx : *(arrayData.mArrayValueFlags)) {
                            os << "                [" << idx << "] "
                               << std::endl;
                        }
                    }
                }
            }
        }
    }

    return result;
}

template <typename ValueT>
using ConverterT = typename openvdb::BoolGrid::ValueConverter<ValueT>::Type;

bool compareUntypedGrids(ComparisonResult &resultData,
                         const openvdb::GridBase &firstGrid,
                         const openvdb::GridBase &secondGrid,
                         const ComparisonSettings &settings,
                         const openvdb::MaskGrid::ConstPtr maskGrid)
{
    bool result = false, valid = false;;
    TypeList::foreach([&](auto x) {
        using GridT = ConverterT<decltype(x)>;
        if (firstGrid.isType<GridT>()) {
            valid = true;
            const GridT& firstGridTyped = static_cast<const GridT&>(firstGrid);
            const GridT& secondGridTyped = static_cast<const GridT&>(secondGrid);
            result = compareGrids(resultData, firstGridTyped, secondGridTyped, settings, maskGrid);
        }
    });

    if (!valid) {
        if (firstGrid.isType<openvdb::points::PointDataGrid>()) {
            valid = true;
            const openvdb::points::PointDataGrid& firstGridTyped =
                static_cast<const openvdb::points::PointDataGrid&>(firstGrid);
            const openvdb::points::PointDataGrid& secondGridTyped =
                static_cast<const openvdb::points::PointDataGrid&>(secondGrid);
            result = compareGrids(resultData, firstGridTyped, secondGridTyped, settings, maskGrid);
        }
    }

    if (!valid) {
        OPENVDB_THROW(openvdb::TypeError, "Unsupported grid type: " + firstGrid.valueType());
    }
    return result;
}


}


