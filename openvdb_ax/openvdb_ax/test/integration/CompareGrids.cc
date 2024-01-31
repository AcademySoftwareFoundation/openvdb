// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file test/integration/CompareGrids.cc

#include "CompareGrids.h"

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/Assert.h>

#include <tbb/concurrent_vector.h>

namespace unittest_util
{

using TypeList = openvdb::TypeList<
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
    using Ptr = std::unique_ptr<DiagnosticData>;
    virtual ~DiagnosticData() = default;

    virtual void report(std::ostream&,
        const openvdb::GridBase&,
        const openvdb::GridBase&,
        openvdb::MaskGrid::Accessor&,
        openvdb::MaskGrid::Accessor&) = 0;
};

template <typename NodeT>
struct NodeDD : public DiagnosticData
{
    using Ptr = std::unique_ptr<NodeDD<NodeT>>;
    using GridT = typename openvdb::BoolGrid::ValueConverter<typename NodeT::ValueType>::Type;

    NodeDD(const openvdb::Coord& origin)
        : mOrigin(origin)
        , mValid(true)
        , mBufferSizes(true)
        , mTopologyFlags(true)
        , mVoxelFlags(true)
        , mDescriptorsMatch(true)
        , mAttributeArrayData() {}
    ~NodeDD() override = default;

    bool hasValueFlags() const { return !mVoxelFlags.isOn(); }
    bool hasTopologyFlags() const { return !mTopologyFlags.isOn(); }
    void flagVoxelTopology(const int16_t idx) { mTopologyFlags.setOff(idx); }
    void flagVoxelValue(const int16_t idx) { mVoxelFlags.setOff(idx); }

    inline DiagnosticArrayData&
    getDiagnosticArrayData(const std::string& name)
    {
        if (!mAttributeArrayData) {
            mAttributeArrayData.reset(new std::map<std::string, DiagnosticArrayData>());
        }
        return (*mAttributeArrayData)[name];
    }

    bool hasDiagnosticArrayData() const { return static_cast<bool>(mAttributeArrayData); }

    inline bool
    hasDiagnosticArrayData(const std::string& name) const {
        return (hasDiagnosticArrayData() &&
            mAttributeArrayData->find(name) != mAttributeArrayData->end());
    }

    inline void report(std::ostream& os,
        const openvdb::GridBase& grid1,
        const openvdb::GridBase& grid2,
        openvdb::MaskGrid::Accessor& accessorTopology,
        openvdb::MaskGrid::Accessor& accessorValues) override
    {
        struct Local {
            // flag to string
            static std::string fts(const bool flag) {
                return (flag ? "[SUCCESS]" : "[FAILED]");
            }
        };

        const GridT& g1 = static_cast<const GridT&>(grid1);
        const GridT& g2 = static_cast<const GridT&>(grid2);

        os << "    Coord        : " << mOrigin << std::endl;
        os << "    Both Valid   : " <<  Local::fts(this->mValid) << std::endl;
        if (!this->mValid) {
            const bool second = g1.constTree().template probeConstNode<NodeT>(mOrigin);
            os << "        Missing in " << (second ? "second" : "first")
               << " grid."
               << std::endl;
            return;
        }

        const auto& l1 = g1.constTree().template probeConstNode<NodeT>(mOrigin);
        const auto& l2 = g2.constTree().template probeConstNode<NodeT>(mOrigin);
        OPENVDB_ASSERT(l1 && l2);

        os << "    Buffer Sizes : " <<  Local::fts(this->mBufferSizes) << std::endl;

        const bool topologyMatch = !this->hasTopologyFlags();
        os << "    Topology     : " <<  Local::fts(topologyMatch) << std::endl;

        if (!topologyMatch) {
            os << "        The following voxel topologies differ : " << std::endl;
            for (auto iter = this->mTopologyFlags.beginOff(); iter; ++iter) {
                const openvdb::Coord coord = l1->offsetToGlobalCoord(iter.pos());
                os << "            [" << iter.pos() << "] "<< coord
                   << " G1: " << l1->isValueOn(coord)
                   << " - G2: " << l2->isValueOn(coord)
                   << std::endl;
                accessorTopology.setValue(coord, true);
            }
        }

        const bool valueMatch = !this->hasValueFlags();
        os << "    Values       : " <<  Local::fts(valueMatch) << std::endl;

        if (!valueMatch) {
            os << "        The following voxel values differ : " << std::endl;
            for (auto iter = this->mVoxelFlags.beginOff(); iter; ++iter) {
                const openvdb::Coord coord = l1->offsetToGlobalCoord(iter.pos());
                os << "            [" << iter.pos() << "] "<< coord
                   << " G1: " << l1->getValue(coord)
                   << " - G2: " << l2->getValue(coord)
                   << std::endl;
                accessorValues.setValue(coord, true);
            }
        }

        if (g1.template isType<openvdb::points::PointDataGrid>()) {
            os << "    Descriptors  : " <<  Local::fts(this->mDescriptorsMatch) << std::endl;
            const bool attributesMatch = !static_cast<bool>(this->mAttributeArrayData);
            os << "    Array Data   : " <<  Local::fts(attributesMatch) << std::endl;
            if (!attributesMatch) {
                os << "        The following attribute values : " << std::endl;
                for (const auto& iter : *(this->mAttributeArrayData)) {

                    const std::string& name = iter.first;
                    const DiagnosticArrayData& arrayData = iter.second;

                    os << "            Attribute Array : [" << name << "] \n"
                       << "            Size Match      : " << Local::fts(arrayData.mSizeMatch) << "\n"
                       << "            Type Match      : " << Local::fts(arrayData.mTypesMatch) << "\n"
                       << "            Flags Match     : " << Local::fts(arrayData.mFlagsMatch)
                       << std::endl;

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

    openvdb::Coord mOrigin;
    bool mValid;
    bool mBufferSizes;
    typename NodeT::NodeMaskType mTopologyFlags;
    typename NodeT::NodeMaskType mVoxelFlags;
    bool mDescriptorsMatch;
    std::unique_ptr<std::map<std::string, DiagnosticArrayData>> mAttributeArrayData;
};

template <typename NodeT,
    typename std::enable_if<(NodeT::LEVEL == 0)>::type* = nullptr>
inline bool compareNodes(const NodeT& firstLeaf,
                         const NodeT& secondLeaf,
                         const typename NodeT::NodeMaskType& mask,
                         NodeDD<NodeT>& data,
                         const ComparisonSettings& settings,
                         const typename NodeT::ValueType& tolerance)
{
    using BufferT = typename NodeT::Buffer;

    const BufferT& firstBuffer = firstLeaf.buffer();
    const BufferT& secondBuffer = secondLeaf.buffer();

    // if the buffers are not the same size the buffer most likely isn't
    // loaded or allocated

    if (firstBuffer.size() != secondBuffer.size()) {
        data.mBufferSizes = false;
        return false;
    }

    const typename NodeT::NodeMaskType& firstMask = firstLeaf.getValueMask();
    const typename NodeT::NodeMaskType& secondMask = secondLeaf.getValueMask();
    typename NodeT::NodeMaskType::OnIterator iter = mask.beginOn();

    for (; iter; ++iter) {
        const openvdb::Index n = iter.pos();
        OPENVDB_ASSERT(n < firstBuffer.size() && n < secondBuffer.size());

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


template <typename NodeT,
    typename std::enable_if<(NodeT::LEVEL != 0)>::type* = nullptr>
inline bool compareNodes(const NodeT& n1,
                         const NodeT& n2,
                         const typename NodeT::NodeMaskType& mask,
                         NodeDD<NodeT>& data,
                         const ComparisonSettings& settings,
                         const typename NodeT::ValueType& tolerance)
{
    const auto& vmask1 = n1.getValueMask();
    const auto& vmask2 = n2.getValueMask();
    const auto& cmask1 = n1.getChildMask();
    const auto& cmask2 = n2.getChildMask();

    auto* t1 = n1.getTable();
    auto* t2 = n2.getTable();

    for (auto iter = mask.beginOn(); iter; ++iter) {
        const openvdb::Index n = iter.pos();

        if ((vmask1.isOn(n) ^ vmask2.isOn(n)) ||
            (cmask1.isOn(n) ^ cmask2.isOn(n))) {
            data.flagVoxelTopology(static_cast<int16_t>(n));
            continue; // can't check values if topology is different
        }

        if (cmask1.isOn(n) && cmask2.isOn(n)) continue;
        OPENVDB_ASSERT(vmask1.isOn(n) && vmask2.isOn(n));

        if (settings.mCheckBufferValues &&
            !openvdb::math::isApproxEqual(t1[n].getValue(), t2[n].getValue(), tolerance)) {
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
                         NodeDD<openvdb::points::PointDataTree::LeafNodeType>& data)
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
                          NodeDD<openvdb::points::PointDataTree::LeafNodeType>& data)
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
                  NodeDD<LeafNodeType>&,
                  const ComparisonSettings&) {
    return true;
}

template <>
inline bool
compareAttributes<openvdb::points::PointDataTree::LeafNodeType>
    (const openvdb::points::PointDataTree::LeafNodeType& firstLeaf,
     const openvdb::points::PointDataTree::LeafNodeType& secondLeaf,
     NodeDD<openvdb::points::PointDataTree::LeafNodeType>& data,
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
struct CompareNodes
{
    using LeafManagerT = openvdb::tree::LeafManager<const openvdb::MaskTree>;
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ConstGridAccessor = openvdb::tree::ValueAccessor<const TreeType>;

    CompareNodes(tbb::concurrent_vector<DiagnosticData::Ptr>& data,
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

    void operator()(const openvdb::MaskTree::RootNodeType&) const {}

    template <typename MaskNodeT>
    void operator()(MaskNodeT& node) const
    {
        using NodeT = typename MaskNodeT::template ValueConverter<typename TreeType::ValueType>::Type;

        const openvdb::Coord& origin = node.origin();
        const NodeT* const n1 = mFirst.template probeConstNode<NodeT>(origin);
        const NodeT* const n2 = mSecond.template probeConstNode<NodeT>(origin);
        if (n1 == nullptr && n2 == nullptr) return;

        typename NodeDD<NodeT>::Ptr data(new NodeDD<NodeT>(origin));

        if (static_cast<bool>(n1) ^ static_cast<bool>(n2)) {
           data->mValid = false;
        }
        else {
            OPENVDB_ASSERT(n1 && n2);
            const typename MaskNodeT::NodeMaskType
                mask(mUseVoxelMask ? node.getValueMask() : true);
            if (compareNodes(*n1, *n2, mask, *data, mSettings, mTolerance) &&
                compareAttributes(*n1, *n2, *data, mSettings)) {
                data.reset();
            }
        }

        if (data) mDiagnosticData.emplace_back(std::move(data));
    }

private:
    tbb::concurrent_vector<DiagnosticData::Ptr>&   mDiagnosticData;
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
    using NodeManager = openvdb::tree::NodeManager<const openvdb::MaskTree,
        openvdb::MaskTree::RootNodeType::LEVEL>;

    struct Local {
        // flag to string
        static std::string fts(const bool flag) {
            return (flag ? "[SUCCESS]" : "[FAILED]");
        }
    };

    bool result = true;
    bool flag = true;
    std::ostream& os = resultData.mOs;

    os << "[Diagnostic : Compare Leaf Nodes Result]\n"
       << "    First Grid: \"" << firstGrid.getName() << "\"\n"
       << "    Second Grid: \"" << secondGrid.getName() << "\"\n"
       << std::endl;

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

    NodeManager manager(mask->constTree());
    tbb::concurrent_vector<DiagnosticData::Ptr> data;

    CompareNodes<TreeType>
        op(data,
           firstGrid.constTree(),
           secondGrid.constTree(),
           tolerance,
           settings);

    manager.foreachBottomUp(op);

    flag = data.empty();
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

    os << "[Diagnostic]: Leaf Node Diagnostics:\n"  << std::endl;

    for (const auto& diag : data) {
        OPENVDB_ASSERT(diag);
        diag->report(os, firstGrid, secondGrid, accessorTopology, accessorValues);
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


