// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/util/Formats.h>

#include "Tree.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

////////////////////////////////////////


template<typename RootNodeType>
void
Tree<RootNodeType>::readTopology(std::istream& is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    TreeBase::readTopology(is, saveFloatAsHalf);
    mRoot.readTopology(is, saveFloatAsHalf);
}


template<typename RootNodeType>
void
Tree<RootNodeType>::writeTopology(std::ostream& os, bool saveFloatAsHalf) const
{
    TreeBase::writeTopology(os, saveFloatAsHalf);
    mRoot.writeTopology(os, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readBuffers(std::istream &is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    mRoot.readBuffers(is, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readBuffers(std::istream &is, const CoordBBox& bbox, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    mRoot.readBuffers(is, bbox, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readNonresidentBuffers() const
{
    for (LeafCIter it = this->cbeginLeaf(); it; ++it) {
        // Retrieving the value of a leaf voxel forces loading of the leaf node's voxel buffer.
        it->getValue(Index(0));
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::writeBuffers(std::ostream &os, bool saveFloatAsHalf) const
{
    mRoot.writeBuffers(os, saveFloatAsHalf);
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clip(const CoordBBox& bbox)
{
    this->clearAllAccessors();
    return mRoot.clip(bbox);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clipUnallocatedNodes()
{
    this->clearAllAccessors();
    for (LeafIter it = this->beginLeaf(); it; ) {
        const LeafNodeType* leaf = it.getLeaf();
        ++it; // advance the iterator before deleting the leaf node
        if (!leaf->isAllocated()) {
            this->addTile(/*level=*/0, leaf->origin(), this->background(), /*active=*/false);
        }
    }
}

template<typename RootNodeType>
inline Index32
Tree<RootNodeType>::unallocatedLeafCount() const
{
    Index32 sum = 0;
    for (auto it = this->cbeginLeaf(); it; ++it) if (!it->isAllocated()) ++sum;
    return sum;
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::sparseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.sparseFill(bbox, value, active);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::denseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.denseFill(bbox, value, active);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::voxelizeActiveTiles(bool threaded)
{
    this->clearAllAccessors();
    mRoot.voxelizeActiveTiles(threaded);
}


template<typename RootNodeType>
Metadata::Ptr
Tree<RootNodeType>::getBackgroundValue() const
{
    Metadata::Ptr result;
    if (Metadata::isRegisteredType(valueType())) {
        using MetadataT = TypedMetadata<ValueType>;
        result = Metadata::createMetadata(valueType());
        if (result->typeName() == MetadataT::staticTypeName()) {
            MetadataT* m = static_cast<MetadataT*>(result.get());
            m->value() = mRoot.background();
        }
    }
    return result;
}


////////////////////////////////////////


// TBB body object used to deallocates nodes in parallel.
template<typename NodeType>
struct DeallocateNodes {
    DeallocateNodes(std::vector<NodeType*>& nodes)
        : mNodes(nodes.empty() ? nullptr : &nodes.front()) { }
    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            delete mNodes[n]; mNodes[n] = nullptr;
        }
    }
    NodeType ** const mNodes;
};


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clear()
{
    std::vector<LeafNodeType*> leafnodes;
    this->stealNodes(leafnodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leafnodes.size()),
        DeallocateNodes<LeafNodeType>(leafnodes));

    std::vector<typename RootNodeType::ChildNodeType*> internalNodes;
    this->stealNodes(internalNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalNodes.size()),
        DeallocateNodes<typename RootNodeType::ChildNodeType>(internalNodes));

    mRoot.clear();

    this->clearAllAccessors();
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<Tree, true>& accessor) const
{
    typename AccessorRegistry::accessor a;
    mAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<const Tree, true>& accessor) const
{
    typename ConstAccessorRegistry::accessor a;
    mConstAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<Tree, true>& accessor) const
{
    mAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<const Tree, true>& accessor) const
{
    mConstAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clearAllAccessors()
{
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }

    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAllAccessors()
{
    mAccessorRegistry.erase(nullptr);
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mAccessorRegistry.clear();

    mAccessorRegistry.erase(nullptr);
    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mConstAccessorRegistry.clear();
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::merge(Tree& other, MergePolicy policy)
{
    this->clearAllAccessors();
    other.clearAllAccessors();
    switch (policy) {
        case MERGE_ACTIVE_STATES:
            mRoot.template merge<MERGE_ACTIVE_STATES>(other.mRoot); break;
        case MERGE_NODES:
            mRoot.template merge<MERGE_NODES>(other.mRoot); break;
        case MERGE_ACTIVE_STATES_AND_NODES:
            mRoot.template merge<MERGE_ACTIVE_STATES_AND_NODES>(other.mRoot); break;
    }
}


template<typename RootNodeType>
Index64
Tree<RootNodeType>::inactiveVoxelCount() const
{
    Coord dim(0, 0, 0);
    this->evalActiveVoxelDim(dim);
    const Index64
        totalVoxels = dim.x() * dim.y() * dim.z(),
        activeVoxels = this->activeVoxelCount();
    assert(totalVoxels >= activeVoxels);
    return totalVoxels - activeVoxels;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafBoundingBox(CoordBBox& bbox) const
{
    bbox.reset(); // default invalid bbox

    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, false);

    return !bbox.empty();
}

template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelBoundingBox(CoordBBox& bbox) const
{
    bbox.reset(); // default invalid bbox

    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, true);

    return !bbox.empty();
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalActiveVoxelBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalLeafBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::evalMinMax(ValueType& minVal, ValueType& maxVal) const
{
    /// @todo optimize
    minVal = maxVal = zeroVal<ValueType>();
    if (ValueOnCIter iter = this->cbeginValueOn()) {
        minVal = maxVal = *iter;
        for (++iter; iter; ++iter) {
            const ValueType& val = *iter;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::getNodeLog2Dims(std::vector<Index>& dims)
{
    dims.clear();
    RootNodeType::getNodeLog2Dims(dims);
}


////////////////////////////////////////


template<typename RootNodeType>
inline const Name&
Tree<RootNodeType>::treeType()
{
    static std::once_flag once;
    std::call_once(once, []()
    {
        std::vector<Index> dims;
        Tree::getNodeLog2Dims(dims);
        std::ostringstream ostr;
        ostr << "Tree_" << typeNameAsString<BuildType>();
        for (size_t i = 1, N = dims.size(); i < N; ++i) { // start from 1 to skip the RootNode
            ostr << "_" << dims[i];
        }
        sTreeTypeName.reset(new Name(ostr.str()));
    });
    return *sTreeTypeName;
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::print(std::ostream& os, int verboseLevel) const
{
    if (verboseLevel <= 0) return;

    /// @todo Consider using boost::io::ios_precision_saver instead.
    struct OnExit {
        std::ostream& os;
        std::streamsize savedPrecision;
        OnExit(std::ostream& _os): os(_os), savedPrecision(os.precision()) {}
        ~OnExit() { os.precision(savedPrecision); }
    };
    OnExit restorePrecision(os);

    std::vector<Index> dims;
    Tree::getNodeLog2Dims(dims);// leaf is the last element

    os << "Information about Tree:\n"
        << "  Type: " << this->type() << "\n";

    os << "  Configuration:\n";

    if (verboseLevel <= 1) {
        // Print node types and sizes.
        os << "    Root(" << mRoot.getTableSize() << ")";
        if (dims.size() > 1) {
            for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
                os << ", Internal(" << (1 << dims[i]) << "^3)";
            }
            os << ", Leaf(" << (1 << dims.back()) << "^3)\n";
        }
        os << "  Background value: " << mRoot.background() << "\n";
        return;
    }

    // The following is tree information that is expensive to extract.

    ValueType minVal = zeroVal<ValueType>(), maxVal = zeroVal<ValueType>();
    if (verboseLevel > 3) {
        // This forces loading of all non-resident nodes.
        this->evalMinMax(minVal, maxVal);
    }

#if OPENVDB_ABI_VERSION_NUMBER >= 7
    const auto nodeCount = this->nodeCount();//fast
    const Index32 leafCount = nodeCount.front();// leaf is the first element
#else
    std::vector<Index64> nodeCount(dims.size());
    for (NodeCIter it = cbeginNode(); it; ++it) ++(nodeCount[it.getDepth()]);//slow
    const Index64 leafCount = *nodeCount.rbegin();// leaf is the last element
#endif
    assert(dims.size() == nodeCount.size());

    Index64 totalNodeCount = 0;
    for (size_t i = 0; i < nodeCount.size(); ++i) totalNodeCount += nodeCount[i];

    // Print node types, counts and sizes.
    os << "    Root(1 x " << mRoot.getTableSize() << ")";
    if (dims.size() >= 2) {
        for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
#if OPENVDB_ABI_VERSION_NUMBER >= 7
            os << ", Internal(" << util::formattedInt(nodeCount[N - i]);
#else
            os << ", Internal(" << util::formattedInt(nodeCount[i]);
#endif
            os << " x " << (1 << dims[i]) << "^3)";
        }
        os << ", Leaf(" << util::formattedInt(leafCount);
        os << " x " << (1 << dims.back()) << "^3)\n";
    }
    os << "  Background value: " << mRoot.background() << "\n";

    // Statistics of topology and values

    if (verboseLevel > 3) {
        os << "  Min value: " << minVal << "\n";
        os << "  Max value: " << maxVal << "\n";
    }

    const Index64
        numActiveVoxels = this->activeVoxelCount(),
        numActiveLeafVoxels = this->activeLeafVoxelCount(),
        numActiveTiles = this->activeTileCount();

    os << "  Number of active voxels:       " << util::formattedInt(numActiveVoxels) << "\n";
    os << "  Number of active tiles:        " << util::formattedInt(numActiveTiles) << "\n";

    Coord dim(0, 0, 0);
    Index64 totalVoxels = 0;
    if (numActiveVoxels) { // nonempty
        CoordBBox bbox;
        this->evalActiveVoxelBoundingBox(bbox);
        dim = bbox.extents();
        totalVoxels = dim.x() * uint64_t(dim.y()) * dim.z();

        os << "  Bounding box of active voxels: " << bbox << "\n";
        os << "  Dimensions of active voxels:   "
            << dim[0] << " x " << dim[1] << " x " << dim[2] << "\n";

        const double activeRatio = (100.0 * double(numActiveVoxels)) / double(totalVoxels);
        os << "  Percentage of active voxels:   " << std::setprecision(3) << activeRatio << "%\n";

        if (leafCount > 0) {
            const double fillRatio = (100.0 * double(numActiveLeafVoxels))
                / (double(leafCount) * double(LeafNodeType::NUM_VOXELS));
            os << "  Average leaf node fill ratio:  " << fillRatio << "%\n";
        }

        if (verboseLevel > 2) {
            Index64 sum = 0;// count the number of unallocated leaf nodes
            for (auto it = this->cbeginLeaf(); it; ++it) if (!it->isAllocated()) ++sum;
            os << "  Number of unallocated nodes:   "
               << util::formattedInt(sum) << " ("
               << (100.0 * double(sum) / double(totalNodeCount)) << "%)\n";
        }
    } else {
        os << "  Tree is empty!\n";
    }
    os << std::flush;

    if (verboseLevel == 2) return;

    // Memory footprint in bytes
    const Index64
        actualMem = this->memUsage(),
        denseMem = sizeof(ValueType) * totalVoxels,
        voxelsMem = sizeof(ValueType) * numActiveLeafVoxels;
            ///< @todo not accurate for BoolTree (and probably should count tile values)

    os << "Memory footprint:\n";
    util::printBytes(os, actualMem, "  Actual:             ");
    util::printBytes(os, voxelsMem, "  Active leaf voxels: ");

    if (numActiveVoxels) {
        util::printBytes(os, denseMem, "  Dense equivalent:   ");
        os << "  Actual footprint is " << (100.0 * double(actualMem) / double(denseMem))
            << "% of an equivalent dense volume\n";
        os << "  Leaf voxel footprint is " << (100.0 * double(voxelsMem) / double(actualMem))
           << "% of actual footprint\n";
    }
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
