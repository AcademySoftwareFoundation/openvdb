// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file OpenToNanoVDB.h

    \author Ken Museth

    \date January 8, 2020

    \brief This class will serialize an OpenVDB grid into a NanoVDB grid.
*/

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/CpuTimer.h>

#include "GridHandle.h" // manages and streams the raw memory buffer of a NanoVDB grid.
#include "GridChecksum.h" // for nanovdb::checksum
#include "GridStats.h" // for nanovdb::Extrema
#include "GridBuilder.h" // for nanovdb::AbsDiff
#include "ForEach.h"// for nanovdb::forEach
#include "Reduce.h"// for nanovdb::reduce
#include "Invoke.h"// for nanovdb::invoke
#include "DitherLUT.h"// for nanovdb::DitherLUT

#include <type_traits>

#ifndef NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED

namespace nanovdb {

/// @brief Converts OpenVDB types to NanoVDB types, e.g. openvdb::Vec3f to nanovdb::Vec3f
///        Template specializations are defined below.
template<typename T>
struct OpenToNanoType { using Type = T; };

//================================================================================================

/// @brief Forward declaration of free-standing function that converts an OpenVDB GridBase into a NanoVDB GridHandle
template<typename BufferT = HostBuffer>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base,
              StatsMode                     sMode = StatsMode::Default,
              ChecksumMode                  cMode = ChecksumMode::Default,
              int                           verbose = 0);

//================================================================================================

/// @brief Forward declaration of free-standing function that converts a typed OpenVDB Grid into a NanoVDB GridHandle
///
/// @details Unlike the function above that takes a base openvdb grid, this method is strongly typed and allows
///          for compression, e.g. openToNanoVDB<HostBuffer, openvdb::FloatTree, nanovdb::Fp16>
template<typename BufferT    = HostBuffer,
         typename OpenTreeT  = openvdb::FloatTree,//dummy default type - it will be resolved from the grid argument
         typename NanoBuildT = typename OpenToNanoType<typename OpenTreeT::BuildType>::Type>
GridHandle<BufferT>
openToNanoVDB(const openvdb::Grid<OpenTreeT>& grid,
              StatsMode                       sMode = StatsMode::Default,
              ChecksumMode                    cMode = ChecksumMode::Default,
              int                             verbose = 0);

//================================================================================================

/// @brief Template specialization for openvdb::Coord
template<>
struct OpenToNanoType<openvdb::Coord>
{
    using Type = nanovdb::Coord;
    static_assert(sizeof(Type) == sizeof(openvdb::Coord), "Mismatching sizeof");
};

/// @brief Template specialization for openvdb::CoordBBox
template<>
struct OpenToNanoType<openvdb::CoordBBox>
{
    using Type = nanovdb::CoordBBox;
    static_assert(sizeof(Type) == sizeof(openvdb::CoordBBox), "Mismatching sizeof");
};

/// @brief Template specialization for openvdb::math::BBox
template<typename T>
struct OpenToNanoType<openvdb::math::BBox<T>>
{
    using Type = nanovdb::BBox<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::BBox<T>), "Mismatching sizeof");
};

/// @brief Template specialization for openvdb::math::Vec3
template<typename T>
struct OpenToNanoType<openvdb::math::Vec3<T>>
{
    using Type = nanovdb::Vec3<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::Vec3<T>), "Mismatching sizeof");
};

/// @brief Template specialization for openvdb::math::Vec4
template<typename T>
struct OpenToNanoType<openvdb::math::Vec4<T>>
{
    using Type = nanovdb::Vec4<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::Vec4<T>), "Mismatching sizeof");
};

/// @brief Template specialization for openvdb::ValueMask
template<>
struct OpenToNanoType<openvdb::ValueMask>
{
    using Type = nanovdb::ValueMask;
};

//================================================================================================

/// @brief Grid trait that defines OpenVDB grids with the exact same configuration as NanoVDB grids
template <typename BuildT>
struct OpenGridType
{
    using GridT  = openvdb::Grid<typename openvdb::tree::Tree4<BuildT, 5, 4, 3>::Type>;
    using TreeT  = typename GridT::TreeType;
    using RootT  = typename TreeT::RootNodeType;
    using UpperT = typename RootT::ChildNodeType;
    using LowerT = typename UpperT::ChildNodeType;
    using LeafT  = typename LowerT::ChildNodeType;
    using ValueT = typename LeafT::ValueType;
};

/// @brief Template specialization for the PointIndexGrid
template <>
struct OpenGridType<openvdb::PointIndex32>
{
    using GridT  = openvdb::tools::PointIndexGrid;// 5, 4, 3
    using TreeT  = typename GridT::TreeType;
    using RootT  = typename TreeT::RootNodeType;
    using UpperT = typename RootT::ChildNodeType;
    using LowerT = typename UpperT::ChildNodeType;
    using LeafT  = typename LowerT::ChildNodeType;
    using ValueT = typename LeafT::ValueType;
};

/// @brief Template specialization for the PointDataGrid
template <>
struct OpenGridType<openvdb::PointDataIndex32>
{
    using GridT  = openvdb::points::PointDataGrid;// 5, 4, 3
    using TreeT  = typename GridT::TreeType;
    using RootT  = typename TreeT::RootNodeType;
    using UpperT = typename RootT::ChildNodeType;
    using LowerT = typename UpperT::ChildNodeType;
    using LeafT  = typename LowerT::ChildNodeType;
    using ValueT = typename LeafT::ValueType;
};

//================================================================================================

/// @brief This class will convert an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
///
/// @note Note that this converter assumes a 5,4,3 tree configuration of BOTH the OpenVDB and NanoVDB
///       grids. This is a consequence of the fact that the OpenVDB tree is defined in OpenGridType and
///       that all NanoVDB trees are by design always 5,4,3!
///
/// @details While NanoVDB allows root, internal and leaf nodes to reside anywhere in the memory buffer
///          this conversion tool uses the following memory layout:
///
///
///  Grid | Tree Root... Node2... Node1... Leaf... BlindMetaData... BlindData...
///  where "..." means size may vary and "|" means "no gap"

template<typename OpenBuildT,
         typename NanoBuildT,
         typename OracleT = AbsDiff,
         typename BufferT = HostBuffer>
class OpenToNanoVDB
{
    struct BlindMetaData; // forward declerations
    template <typename NodeT> struct NodePair;
    struct Codec {float min, max; uint16_t log2, size;};// used for adaptive bit-rate quantization

    using OpenGridT = typename OpenGridType<OpenBuildT>::GridT;//   OpenVDB grid
    using OpenTreeT = typename OpenGridType<OpenBuildT>::TreeT;//   OpenVDB tree
    using OpenRootT = typename OpenGridType<OpenBuildT>::RootT;//   OpenVDB root node
    using OpenUpperT= typename OpenGridType<OpenBuildT>::UpperT;//  OpenVDB upper internal node
    using OpenLowerT= typename OpenGridType<OpenBuildT>::LowerT;//  OpenVDB lower internal node
    using OpenLeafT = typename OpenGridType<OpenBuildT>::LeafT;//   OpenVDB leaf node
    using OpenValueT= typename OpenGridType<OpenBuildT>::ValueT;

    using NanoValueT= typename BuildToValueMap<NanoBuildT>::Type;// e.g. maps from Fp16 to float
    using NanoLeafT = NanoLeaf<NanoBuildT>;
    using NanoLowerT= NanoLower<NanoBuildT>;
    using NanoUpperT= NanoUpper<NanoBuildT>;
    using NanoRootT = NanoRoot<NanoBuildT>;
    using NanoTreeT = NanoTree<NanoBuildT>;
    using NanoGridT = NanoGrid<NanoBuildT>;

    static_assert(sizeof(NanoValueT) == sizeof(OpenValueT), "Mismatching sizeof");
    static_assert(is_same<NanoValueT, typename OpenToNanoType<OpenValueT>::Type>::value, "Mismatching ValueT");

    NanoValueT                        mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    uint8_t*                          mBufferPtr;// pointer to the beginning of the buffer
    uint64_t                          mBufferOffsets[9];//grid, tree, root, upper. lower, leafs, meta data, blind data, buffer size
    int                               mVerbose;
    std::set<BlindMetaData>           mBlindMetaData; // sorted according to index
    std::vector<NodePair<OpenLeafT >> mArray0; // leaf nodes
    std::vector<NodePair<OpenLowerT>> mArray1; // lower internal nodes
    std::vector<NodePair<OpenUpperT>> mArray2; // upper internal nodes
    std::unique_ptr<Codec[]>          mCodec;// defines a codec per leaf node
    StatsMode                         mStats;
    ChecksumMode                      mChecksum;
    bool                              mDitherOn;
    OracleT                           mOracle;// used for adaptive bit-rate quantization

public:
    /// @brief Default c-tor
    OpenToNanoVDB();

    /// @brief return a reference to the compression oracle
    ///
    /// @note Note, the oracle is only used when NanoBuildT = nanovdb::FpN!
    OracleT& oracle() { return mOracle; }

    void setVerbose(int mode = 1) { mVerbose = mode; }

    void enableDithering(bool on = true) { mDitherOn = on; }

    void setStats(StatsMode mode = StatsMode::Default) { mStats = mode; }

    void setChecksum(ChecksumMode mode = ChecksumMode::Default) { mChecksum = mode; }

    /// @brief Return a shared pointer to a NanoVDB grid handle constructed from the specified OpenVDB grid
    GridHandle<BufferT> operator()(const OpenGridT& grid,
                                   const BufferT&   allocator = BufferT());

    GridHandle<BufferT> operator()(const OpenGridT& grid,
                                   StatsMode        sMode,
                                   ChecksumMode     cMode,
                                   int              verbose,
                                   const BufferT&   allocator = BufferT());

private:

    /// @brief Allocates and return a handle for the buffer
    GridHandle<BufferT> initHandle(const OpenGridT& openGrid, const BufferT& allocator);

    template <typename T>
    inline typename std::enable_if<!std::is_same<T, FpN>::value>::type
    compression(const OpenGridT&, uint64_t&) {}// no-op

    template <typename T>
    inline typename std::enable_if<std::is_same<T, FpN>::value>::type
    compression(const OpenGridT& openGrid, uint64_t &offset);

    /// @brief Private method to process the grid
    NanoGridT* processGrid(const OpenGridT& openGrid);

    // @brief Private method to process the tree
    NanoTreeT* processTree(const OpenTreeT& openTree);

    /// @brief Private method to process the root node
    NanoRootT* processRoot(const OpenRootT& openRoot);

    template <typename T>
    void processNodes(std::vector<NodePair<T>> &nodes);

    //////////////////////

    template<typename T>
    typename std::enable_if<!std::is_same<typename OpenGridType<openvdb::ValueMask>::LeafT, typename T::OpenNodeT>::value &&
                            !std::is_same<typename OpenGridType<bool>::LeafT, typename T::OpenNodeT>::value &&
                            !std::is_same<Fp4, typename T::NanoNodeT::BuildType>::value &&
                            !std::is_same<Fp8, typename T::NanoNodeT::BuildType>::value &&
                            !std::is_same<Fp16,typename T::NanoNodeT::BuildType>::value &&
                            !std::is_same<FpN, typename T::NanoNodeT::BuildType>::value>::type
    processLeafs(std::vector<T> &leafs);

    template<typename T>
    typename std::enable_if<std::is_same<Fp4,  typename T::NanoNodeT::BuildType>::value ||
                            std::is_same<Fp8,  typename T::NanoNodeT::BuildType>::value ||
                            std::is_same<Fp16, typename T::NanoNodeT::BuildType>::value>::type
    processLeafs(std::vector<T> &leafs);

    template<typename T>
    typename std::enable_if<std::is_same<FpN,  typename T::NanoNodeT::BuildType>::value>::type
    processLeafs(std::vector<T> &leafs);

    template<typename T>
    typename std::enable_if<std::is_same<T, typename OpenGridType<openvdb::ValueMask>::LeafT>::value>::type
    processLeafs(std::vector<NodePair<T>> &leafs);

    template<typename T>
    typename std::enable_if<std::is_same<T, typename OpenGridType<bool>::LeafT>::value>::type
    processLeafs(std::vector<NodePair<T>> &leafs);

    //////////////////////

    /// @brief Private methods to pre-process the bind metadata
    template <typename T>
    typename std::enable_if<!std::is_same<T, openvdb::tools::PointIndexGrid>::value &&
                            !std::is_same<T, openvdb::points::PointDataGrid>::value>::type
    preProcessMetadata(const T& openGrid);

    template <typename T>
    typename std::enable_if<std::is_same<T, openvdb::tools::PointIndexGrid>::value>::type
    preProcessMetadata(const T& openGrid);

    template <typename T>
    typename std::enable_if<std::is_same<T, openvdb::points::PointDataGrid>::value>::type
    preProcessMetadata(const T& openGrid);

    //////////////////////

    /// @brief Private methods to process the blind metadata
    template<typename T>
    typename std::enable_if<!std::is_same<T, openvdb::tools::PointIndexGrid>::value &&
                            !std::is_same<T, openvdb::points::PointDataGrid>::value, GridBlindMetaData*>::type
    processMetadata(const T& openGrid);

    template<typename T>
    typename std::enable_if<std::is_same<T, openvdb::tools::PointIndexGrid>::value, GridBlindMetaData*>::type
    processMetadata(const T& openGrid);

    template<typename T>
    typename std::enable_if<std::is_same<T, openvdb::points::PointDataGrid>::value, GridBlindMetaData*>::type
    processMetadata(const T& openGrid);

    //////////////////////

    uint64_t pointCount();

    template<typename AttT, typename CodecT = openvdb::points::UnknownCodec>
    void copyPointAttribute(size_t attIdx, AttT *attPtr);

    /// @brief Performs: nanoNode.origin = openNode.origin
    ///                  openNode.origin = nanoNode offset
    template <typename OpenNodeT, typename NanoNodeT>
    void encode(const OpenNodeT *openNode, NanoNodeT *nanoNode);

    /// @brief Performs: nanoNode offset = openNode.origin
    ///                  openNode.origin = nanoNode.origin
    ///                  return nanoNode offset
    template <typename OpenNodeT>
    typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type* decode(const OpenNodeT *openNode);

}; // OpenToNanoVDB class

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::OpenToNanoVDB()
    : mVerbose(0)
    , mStats(StatsMode::Default)
    , mChecksum(ChecksumMode::Default)
    , mDitherOn(false)
    , mOracle()
{
}

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
inline GridHandle<BufferT>
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    operator()(const OpenGridT& openGrid,
               StatsMode        sMode,
               ChecksumMode     cMode,
               int              verbose,
               const BufferT&   allocator)
{
    this->setStats(sMode);
    this->setChecksum(cMode);
    this->setVerbose(verbose);
    return (*this)(openGrid, allocator);
}

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
inline GridHandle<BufferT>
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    operator()(const OpenGridT& openGrid,
               const BufferT&   allocator)
{
    std::unique_ptr<openvdb::util::CpuTimer> timer(mVerbose > 1 ? new openvdb::util::CpuTimer() : nullptr);

    if (timer) timer->start("Allocating memory for the NanoVDB buffer");
    auto handle = this->initHandle(openGrid, allocator);
    if (timer) timer->stop();

    if (timer) timer->start("Processing leaf nodes");
    this->processLeafs(mArray0);
    if (timer) timer->stop();

    if (timer) timer->start("Processing lower internal nodes");
    this->processNodes(mArray1);
    if (timer) timer->stop();

    if (timer) timer->start("Processing upper internal nodes");
    this->processNodes(mArray2);
    if (timer) timer->stop();

    if (timer) timer->start("Processing grid, tree and root node");
    NanoGridT *nanoGrid = this->processGrid(openGrid);
    if (timer) timer->stop();

    // Point grids already make use of min/max so they shouldn't be re-computed
    if (std::is_same<OpenBuildT, openvdb::PointIndex32>::value ||
        std::is_same<OpenBuildT, openvdb::PointDataIndex32>::value) {
        if (mStats > StatsMode::BBox) mStats = StatsMode::BBox;
    }

    if (timer) timer->start("GridStats");
    gridStats(*nanoGrid, mStats);
    if (timer) timer->stop();

    if (timer) timer->start("Checksum");
    updateChecksum(*nanoGrid, mChecksum);
    if (timer) timer->stop();

    return handle; // invokes move constructor
} // OpenToNanoVDB::operator()

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename T>
inline typename std::enable_if<std::is_same<T, FpN>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    compression(const OpenGridT& openGrid, uint64_t &offset)
{
    static_assert(is_same<float, OpenBuildT>::value, "compression: expected OpenBuildT == float");
    static_assert(is_same<FpN,   NanoBuildT>::value, "compression: expected NanoBuildT == FpN");
    if (is_same<AbsDiff, OracleT>::value && mOracle.getTolerance() < 0.0f) {// default tolerance for level set and fog volumes
        if (openGrid.getGridClass() == openvdb::GRID_LEVEL_SET) {
            mOracle.setTolerance(0.1f * openGrid.voxelSize()[0]);// range of ls: [-3dx; 3dx]
        } else if (openGrid.getGridClass() == openvdb::GRID_FOG_VOLUME) {
            mOracle.setTolerance(0.01f);// range of FOG volumes: [0;1]
        } else {
            mOracle.setTolerance(0.0f);
        }
    }

    const size_t size = mArray0.size();
    mCodec.reset(new Codec[size]);

    DitherLUT lut(mDitherOn);
    auto kernel = [&](const auto &r) {
        const OracleT oracle = mOracle;
        for (auto i=r.begin(); i!=r.end(); ++i) {
            const float *data = mArray0[i].node->buffer().data();
            float min = std::numeric_limits<float>::max(), max = -min;
            for (int j=0; j<512; ++j) {
                float v = data[j];
                if (v<min) min=v;
                if (v>max) max=v;
            }
            mCodec[i].min = min;
            mCodec[i].max = max;
            const float range = max - min;
            uint16_t logBitWidth = 0;// 0,1,2,3,4 => 1,2,4,8,16 bits
            while (range > 0.0f && logBitWidth < 4u) {
                const uint32_t mask = (uint32_t(1) << (uint32_t(1) << logBitWidth)) - 1u;
                const float encode  = mask/range;
                const float decode  = range/mask;
                int j = 0;
                do {
                    const float exact  = data[j];// exact value
                    const uint32_t code = uint32_t(encode*(exact - min) + lut(j));
                    const float approx = code * decode + min;// approximate value
                    j += mOracle(exact, approx) ? 1 : 513;
                } while(j < 512);
                if (j == 512) break;
                ++logBitWidth;
            }
            mCodec[i].log2 = logBitWidth;
            mCodec[i].size = NanoLeafT::DataType::memUsage(1u<<logBitWidth);
        }
    };// kernel
    forEach(0, size, 4, kernel);

    if (mVerbose) {
        uint32_t counters[5+1] = {0};
        ++counters[mCodec[0].log2];
        for (size_t i=1; i<size; ++i) {
            ++counters[mCodec[i].log2];
            mArray0[i].offset = mArray0[i-1].offset + mCodec[i-1].size;
        }
        std::cout << "\n" << mOracle << std::endl;
        std::cout << "Dithering: " << (mDitherOn ? "enabled" : "disabled") << std::endl;
        float avg = 0.0f;
        for (uint32_t i=0; i<=5; ++i) {
            if (uint32_t n = counters[i]) {
                avg += n * float(1 << i);
                printf("%2i bits: %6u leaf nodes, i.e. %4.1f%%\n",1<<i, n, 100.0f*n/float(size));
            }
        }
        printf("%4.1f bits per value on average\n", avg/float(size));
    } else {
        for (size_t i=1; i<size; ++i) {
            mArray0[i].offset = mArray0[i-1].offset + mCodec[i-1].size;
        }
    }
    offset = mArray0[size-1].offset + mCodec[size-1].size;
}// OpenToNanoVDB::compression

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
GridHandle<BufferT> OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    initHandle(const OpenGridT& openGrid, const BufferT& buffer)
{
    auto &openTree = openGrid.tree();
    auto &openRoot = openTree.root();

    mArray0.clear();
    mArray1.clear();
    mArray2.clear();
    std::vector<uint32_t> nodeCount = openTree.nodeCount();
    mArray0.reserve(nodeCount[0]);
    mArray1.reserve(nodeCount[1]);
    mArray2.reserve(nodeCount[2]);

    uint64_t offset[3] = {0};
    for (auto it2 = openRoot.cbeginChildOn(); it2; ++it2) {
        mArray2.emplace_back(&(*it2), offset[2]);
        offset[2] += NanoUpperT::memUsage();
        for (auto it1 = it2->cbeginChildOn(); it1; ++it1) {
            mArray1.emplace_back(&(*it1), offset[1]);
            offset[1] += NanoLowerT::memUsage();
            for (auto it0 = it1->cbeginChildOn(); it0; ++it0) {
                mArray0.emplace_back(&(*it0), offset[0]);
                offset[0] += sizeof(NanoLeafT);
            }
        }
    }

    this->template compression<NanoBuildT>(openGrid, offset[0]);

    this->preProcessMetadata(openGrid);

    mBufferOffsets[0] = 0;// grid is always plated at the beginning of the buffer!
    mBufferOffsets[1] = NanoGridT::memUsage(); // grid ends and tree begins
    mBufferOffsets[2] = NanoTreeT::memUsage(); // tree ends and root begins
    mBufferOffsets[3] = NanoRootT::memUsage(openTree.root().getTableSize()); // root ends and upper internal nodes begins
    mBufferOffsets[4] = offset[2];// upper ends and lower internal nodes
    mBufferOffsets[5] = offset[1];// lower ends and leaf nodes begins
    mBufferOffsets[6] = offset[0];// leafs end blind meta data begins
    mBufferOffsets[7] = GridBlindMetaData::memUsage(mBlindMetaData.size()); // meta ends and blind data begins
    mBufferOffsets[8] = 0;// blind data
    for (auto& i : mBlindMetaData) mBufferOffsets[8] += i.size; // blind data

    // Compute the prefixed sum
    for (int i = 2; i < 9; ++i) {
        mBufferOffsets[i] += mBufferOffsets[i - 1];
    }

    GridHandle<BufferT> handle(BufferT::create(mBufferOffsets[8], &buffer));
    mBufferPtr = handle.data();

    if (mVerbose) {
        openvdb::util::printBytes(std::cout, mBufferOffsets[8], "Allocated", " for the NanoVDB grid\n");
    }
    return handle;// is converted to r-value so return value is move constructed!
}// OpenToNanoVDB::initHandle

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
NanoGrid<NanoBuildT>* OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    processGrid(const OpenGridT& openGrid)
{
    auto *nanoGrid = reinterpret_cast<NanoGridT*>(mBufferPtr + mBufferOffsets[0]);
    if (!openGrid.transform().baseMap()->isLinear()) {
        OPENVDB_THROW(openvdb::ValueError, "processGrid: OpenToNanoVDB only supports grids with affine transforms");
    }
    auto  affineMap = openGrid.transform().baseMap()->getAffineMap();
    auto *data = nanoGrid->data();
    data->mMagic = NANOVDB_MAGIC_NUMBER;
    data->mChecksum = 0u;
    data->mVersion = Version();
    data->mFlags = static_cast<uint32_t>(GridFlags::IsBreadthFirst);
    data->mGridIndex = 0;
    data->mGridCount = 1;
    data->mGridSize = mBufferOffsets[8];
    data->mWorldBBox = BBox<Vec3R>();
    data->mBlindMetadataOffset = 0;
    data->mBlindMetadataCount = 0;

    const std::string gridName = openGrid.getName();
    strncpy(data->mGridName, gridName.c_str(), GridData::MaxNameSize-1);
    data->mGridName[GridData::MaxNameSize-1] ='\0';// null terminate
    if (gridName.length() >= GridData::MaxNameSize) {
        data->setLongGridNameOn();// grid name is long so store it as blind data
    }
    mDelta = NanoValueT(0); // dummy value
    switch (openGrid.getGridClass()) { // set grid class
    case openvdb::GRID_LEVEL_SET:
        if (!is_floating_point<OpenValueT>::value)
            OPENVDB_THROW(openvdb::ValueError, "processGrid: Level sets are expected to be floating point types");
        data->mGridClass = GridClass::LevelSet;
        mDelta = NanoValueT(openGrid.voxelSize()[0]); // skip a node if max < -mDelta || min > mDelta
        break;
    case openvdb::GRID_FOG_VOLUME:
        data->mGridClass = GridClass::FogVolume;
        break;
    case openvdb::GRID_STAGGERED:
        data->mGridClass = GridClass::Staggered;
        break;
    default:
        data->mGridClass = GridClass::Unknown;
    }

    // mapping from the OpenVDB build type to the NanoVDB build type and GridType enum
    if (std::is_same<NanoBuildT, float>::value) { // resolved at compiletime
        data->mGridType = GridType::Float;
    } else if (std::is_same<NanoBuildT, double>::value) {
        data->mGridType = GridType::Double;
    } else if (std::is_same<NanoBuildT, int16_t>::value) {
        data->mGridType = GridType::Int16;
    } else if (std::is_same<NanoBuildT, int32_t>::value) {
        data->mGridType = GridType::Int32;
    } else if (std::is_same<NanoBuildT, int64_t>::value) {
        data->mGridType = GridType::Int64;
    } else if (std::is_same<NanoBuildT, Vec3f>::value) {
        data->mGridType = GridType::Vec3f;
    } else if (std::is_same<NanoBuildT, openvdb::Index32>::value) {
        data->mGridType = GridType::UInt32;
    } else if (std::is_same<NanoBuildT, openvdb::PointIndex32>::value) {
        data->mGridType = GridType::UInt32;
        data->mGridClass = GridClass::PointIndex;
    } else if (std::is_same<NanoBuildT, openvdb::PointDataIndex32>::value) {
        data->mGridType = GridType::UInt32;
        data->mGridClass = GridClass::PointData;
    } else if (std::is_same<NanoBuildT, ValueMask>::value) {
        data->mGridType = GridType::Mask;
        data->mGridClass = GridClass::Topology;
    } else if (std::is_same<NanoBuildT, bool>::value) {
        data->mGridType = GridType::Boolean;
    } else if (std::is_same<NanoBuildT, Fp4>::value) {
        data->mGridType = GridType::Fp4;
    } else if (std::is_same<NanoBuildT, Fp8>::value) {
        data->mGridType = GridType::Fp8;
    } else if (std::is_same<NanoBuildT, Fp16>::value) {
        data->mGridType = GridType::Fp16;
    } else if (std::is_same<NanoBuildT, FpN>::value) {
        data->mGridType = GridType::FpN;
    } else if (std::is_same<NanoBuildT, Vec4f>::value) {
        data->mGridType = GridType::Vec4f;
    } else if (std::is_same<NanoBuildT, Vec4d>::value) {
        data->mGridType = GridType::Vec4d;
    } else {
        OPENVDB_THROW(openvdb::ValueError, "processGrid: Unsupported value type");
    }
    { // set affine map
        if (openGrid.hasUniformVoxels()) {
            data->mVoxelSize = nanovdb::Vec3R(affineMap->voxelSize()[0]);
        } else {
            data->mVoxelSize = affineMap->voxelSize();
        }
        const auto mat = affineMap->getMat4();
        // Only support non-tapered at the moment:
        data->mMap.set(mat, mat.inverse(), 1.0);
    }

    this->processTree(openGrid.tree());// calls processRoot

    if (auto size = mBlindMetaData.size()) {
        auto *metaData = this->processMetadata(openGrid);
        data->mBlindMetadataOffset = PtrDiff(metaData, nanoGrid);
        data->mBlindMetadataCount = static_cast<uint32_t>(size);
        auto *blindData = reinterpret_cast<char*>(mBufferPtr + mBufferOffsets[7]);
        metaData->setBlindData(blindData);
    }
    return nanoGrid;
}// OpenToNanoVDB::processGrid

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
NanoTree<NanoBuildT>* OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    processTree(const OpenTreeT& openTree)
{
    auto *nanoTree = reinterpret_cast<NanoTreeT*>(mBufferPtr + mBufferOffsets[1]);
    auto *data = nanoTree->data();

    data->setRoot( this->processRoot( openTree.root()) );

    NanoUpperT *nanoUpper = mArray2.empty() ? nullptr : reinterpret_cast<NanoUpperT*>(mBufferPtr + mBufferOffsets[3]);
    data->setFirstNode(nanoUpper);

    NanoLowerT *nanoLower = mArray1.empty() ? nullptr : reinterpret_cast<NanoLowerT*>(mBufferPtr + mBufferOffsets[4]);
    data->setFirstNode(nanoLower);

    NanoLeafT  *nanoLeaf  = mArray0.empty() ? nullptr : reinterpret_cast<NanoLeafT*>(mBufferPtr + mBufferOffsets[5]);
    data->setFirstNode(nanoLeaf);

    data->mNodeCount[0] = mArray0.size();
    data->mNodeCount[1] = mArray1.size();
    data->mNodeCount[2] = mArray2.size();

#if 1// count active tiles and voxels

    // Count number of active leaf level tiles
    data->mTileCount[0] = reduce(mArray1, uint32_t(0), [&](auto &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray1[i].node->getValueMask().countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active lower internal node tiles
    data->mTileCount[1] = reduce(mArray2, uint32_t(0), [&](auto &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray2[i].node->getValueMask().countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active upper internal node tiles
    uint32_t sum = 0;
    for (auto it = openTree.root().cbeginValueOn(); it; ++it) ++sum;
    data->mTileCount[2] = sum;

    data->mVoxelCount = reduce(mArray0, uint64_t(0), [&](auto &r, uint64_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray0[i].node->valueMask().countOn();
        return sum;}, std::plus<uint64_t>());

    data->mVoxelCount += data->mTileCount[0]*NanoLeafT::NUM_VALUES;
    data->mVoxelCount += data->mTileCount[1]*NanoLowerT::NUM_VALUES;
    data->mVoxelCount += data->mTileCount[2]*NanoUpperT::NUM_VALUES;

#else

    data->mTileCount[0] = 0;
    data->mTileCount[1] = 0;
    data->mTileCount[2] = 0;
    data->mVoxelCount = 0;

#endif

    return nanoTree;
}// OpenToNanoVDB::processTree

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
NanoRoot<NanoBuildT>* OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    processRoot(const OpenRootT& openRoot)
{
    auto *nanoRoot = reinterpret_cast<NanoRootT*>(mBufferPtr + mBufferOffsets[2]);
    auto* data = nanoRoot->data();
    data->mBackground = openRoot.background();
    data->mTableSize = 0;// incremented below
    data->mMinimum = data->mMaximum = data->mBackground;
    data->mBBox.min() = openvdb::Coord::max(); // set to an empty bounding box
    data->mBBox.max() = openvdb::Coord::min();

    OpenValueT value = openvdb::zeroVal<OpenValueT>();// to avoid compiler warning
    for (auto iter = openRoot.cbeginChildAll(); iter; ++iter) {
        auto* tile = data->tile(data->mTableSize++);
        if (const OpenUpperT *openChild = iter.probeChild( value )) {
            tile->setChild(iter.getCoord(), this->decode(openChild), data);
        } else {
            tile->setValue(iter.getCoord(), iter.isValueOn(), value);
        }
    }
    return nanoRoot;
} // OpenToNanoVDB::processRoot

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename OpenNodeT>
void OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    processNodes(std::vector<NodePair<OpenNodeT>>& openNodes)
{
    using NanoNodeT = typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type;
    static_assert(NanoNodeT::LEVEL == 1 || NanoNodeT::LEVEL == 2, "Expected internal node");
    auto  kernel = [&](const Range1D& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5 - NanoNodeT::LEVEL];// 3 or 4
        OpenValueT value = openvdb::zeroVal<OpenValueT>();// to avoid compiler warning
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openNode = openNodes[i].node;
            auto *nanoNode = PtrAdd<NanoNodeT>(ptr, openNodes[i].offset);
            auto*    data  = nanoNode->data();
            this->encode(openNode, nanoNode);
            data->mValueMask = openNode->getValueMask(); // copy value mask
            data->mChildMask = openNode->getChildMask(); // copy child mask
            for (auto iter = openNode->cbeginChildAll(); iter; ++iter) {
                if (const auto *openChild = iter.probeChild(value)) {
                    data->setChild(iter.pos(), this->decode(openChild));
                } else {
                    data->setValue(iter.pos(), value);
                }
            }
        }
    };
    forEach(openNodes, 1, kernel);
} // OpenToNanoVDB::processNodes

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<!std::is_same<typename OpenGridType<openvdb::ValueMask>::LeafT, typename T::OpenNodeT>::value &&
                               !std::is_same<typename OpenGridType<bool>::LeafT, typename T::OpenNodeT>::value &&
                               !std::is_same<Fp4, typename T::NanoNodeT::BuildType>::value &&
                               !std::is_same<Fp8, typename T::NanoNodeT::BuildType>::value &&
                               !std::is_same<Fp16,typename T::NanoNodeT::BuildType>::value &&
                               !std::is_same<FpN, typename T::NanoNodeT::BuildType>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processLeafs(std::vector<T>& openLeafs)
{
    auto kernel = [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openLeaf = openLeafs[i].node;
            auto *nanoLeaf = PtrAdd<NanoLeafT>(ptr, openLeafs[i].offset);
            auto* data = nanoLeaf->data();
            this->encode(openLeaf, nanoLeaf);
            data->mFlags = 0u;
            data->mValueMask = openLeaf->valueMask(); // copy value mask
            auto *src = reinterpret_cast<const NanoValueT*>(openLeaf->buffer().data());
            for (NanoValueT *dst = data->mValues, *end = dst + OpenLeafT::size(); dst != end; dst += 4, src += 4) {
                dst[0] = src[0]; // copy *all* voxel values in sets of four, i.e. loop-unrolling
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
    };
    forEach(openLeafs, 8, kernel);
} // OpenToNanoVDB::processLeafs<T>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<Fp4,  typename T::NanoNodeT::BuildType>::value ||
                               std::is_same<Fp8,  typename T::NanoNodeT::BuildType>::value ||
                               std::is_same<Fp16, typename T::NanoNodeT::BuildType>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processLeafs(std::vector<T>& openLeafs)
{
    using ArrayT = typename NanoLeafT::DataType::ArrayType;
    using FloatT = typename std::conditional<NanoLeafT::DataType::bitWidth()>=16, double, float>::type;// 16 compression and higher requires double
    DitherLUT lut(mDitherOn);

    auto kernel = [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openLeaf = openLeafs[i].node;
            auto *nanoLeaf = PtrAdd<NanoLeafT>(ptr, openLeafs[i].offset);
            auto* data = nanoLeaf->data();
            this->encode(openLeaf, nanoLeaf);
            data->mFlags = 0u;
            data->mValueMask = openLeaf->valueMask(); // copy value mask
            auto *src = reinterpret_cast<const float*>(openLeaf->buffer().data());
            // compute extrema values
            float min = std::numeric_limits<float>::max(), max = -min;
            for (int i=0; i<512; ++i) {
                const float v = src[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            data->init(min, max, NanoLeafT::DataType::bitWidth());
            // perform quantization relative to the values in the curret leaf node
            const FloatT encode = FloatT((1 << NanoLeafT::DataType::bitWidth()) - 1)/(max-min);
            auto *code = reinterpret_cast<ArrayT*>(data->mCode);
            int offset = 0;
            if (std::is_same<Fp4,  NanoBuildT>::value) {// resolved at compile-time
                for (int i=0; i<128; ++i) {
                    auto tmp = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++  = ArrayT(encode * (*src++ - min) + lut(offset++)) << 4 | tmp;
                    tmp      = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++  = ArrayT(encode * (*src++ - min) + lut(offset++)) << 4 | tmp;
                }
            } else {
                for (int i=0; i<128; ++i) {
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                }
            }
        }
    };
    forEach(openLeafs, 8, kernel);
} // OpenToNanoVDB::processLeafs<Fp4, Fp8, Fp16>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<FpN, typename T::NanoNodeT::BuildType>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processLeafs(std::vector<T>& openLeafs)
{
    static_assert(is_same<float, OpenBuildT>::value, "Expected OpenBuildT == float");

    DitherLUT lut(mDitherOn);
    auto kernel = [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openLeaf = openLeafs[i].node;
            auto *nanoLeaf = PtrAdd<NanoLeafT>(ptr, openLeafs[i].offset);
            auto* data = nanoLeaf->data();
            this->encode(openLeaf, nanoLeaf);
            const uint8_t logBitWidth = uint8_t(mCodec[i].log2);
            data->mFlags = logBitWidth << 5;// pack logBitWidth into 3 MSB of mFlag
            data->mValueMask = openLeaf->valueMask(); // copy value mask
            auto *src = reinterpret_cast<const float*>(openLeaf->buffer().data());
            const float min = mCodec[i].min, max = mCodec[i].max;
            data->init(min, max, uint8_t(1) << logBitWidth);
            // perform quantization relative to the values in the curret leaf node
            int offset = 0;
            switch (logBitWidth) {
                case 0u: {// 1 bit
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 1.0f/(max - min);
                    for (int j=0; j<64; ++j) {
                        uint8_t a = 0;
                        for (int k=0; k<8; ++k) {
                            a |= uint8_t(encode * (*src++ - min) + lut(offset++)) << k;
                        }
                        *dst++ = a;
                    }
                }
                break;
                case 1u: {// 2 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 3.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = uint8_t(encode * (*src++ - min) + lut(offset++));
                        a     |= uint8_t(encode * (*src++ - min) + lut(offset++)) << 2;
                        a     |= uint8_t(encode * (*src++ - min) + lut(offset++)) << 4;
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 6 | a;
                    }
                }
                break;
                case 2u: {// 4 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 15.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 4 | a;
                        a      = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 4 | a;
                    }
                }
                break;
                case 3u: {// 8 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 255.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                    }
                }
                break;
                default: {// 16 bits
                    auto *dst = reinterpret_cast<uint16_t*>(data+1);
                    const double encode = 65535.0/(max - min);// note that double is required!
                    for (int j=0; j<128; ++j) {
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                    }
                }
            }// end switch
        }
    };// kernel
    forEach(openLeafs, 8, kernel);
} // OpenToNanoVDB::processLeafs<FpN>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<T, typename OpenGridType<bool>::LeafT>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processLeafs(std::vector<NodePair<T>>& openLeafs)
{
    auto kernel = [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openLeaf = openLeafs[i].node;
            auto *nanoLeaf = PtrAdd<NanoLeafT>(ptr, openLeafs[i].offset);
            auto* data = nanoLeaf->data();
            this->encode(openLeaf, nanoLeaf);
            data->mFlags = 0u;
            data->mValueMask = openLeaf->valueMask(); // copy value mask
            data->mValues = *reinterpret_cast<const nanovdb::Mask<3>*>(openLeaf->buffer().data()); // copy values
        }
    };
    forEach(openLeafs, 8, kernel);
} // OpenToNanoVDB::processLeafs<bool>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<T, typename OpenGridType<openvdb::ValueMask>::LeafT>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processLeafs(std::vector<NodePair<T>>& openLeafs)
{
    auto kernel = [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *openLeaf = openLeafs[i].node;
            auto *nanoLeaf = PtrAdd<NanoLeafT>(ptr, openLeafs[i].offset);
            auto* data = nanoLeaf->data();
            this->encode(openLeaf, nanoLeaf);
            data->mFlags = 0u;
            data->mValueMask = openLeaf->valueMask(); // copy value mask
        }
    };
    forEach(openLeafs, 8, kernel);
} // OpenToNanoVDB::processLeafs<ValueMask>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
uint64_t OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::pointCount()
{
    return reduce(mArray0, uint64_t(0), [&](auto &r, uint64_t sum) {
           for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray0[i].node->getLastValue();
           return sum;}, std::plus<uint64_t>());
}// OpenToNanoVDB::pointCount

//================================================================================================

/// @brief Performs: nanoNode.origin = openNode.origin
///                  openNode.origin = nanoNode offset
template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename OpenNodeT, typename NanoNodeT>
inline void OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
encode(const OpenNodeT *openNode, NanoNodeT *nanoNode)
{
    static_assert(is_same<NanoNodeT, typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type>::value, "Type mismatch");
    openvdb::Coord &ijk = const_cast<openvdb::Coord&>(openNode->origin());
    nanoNode->data()->setOrigin(ijk);
    reinterpret_cast<int64_t&>(ijk) = PtrDiff(nanoNode, mBufferPtr);
}// OpenToNanoVDB::encode

//================================================================================================

/// @brief Performs: nanoNode offset = openNode.origin
///                  openNode.origin = nanoNode.origin
///                  return nanoNode offset
template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename OpenNodeT>
inline typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type* OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
decode(const OpenNodeT *openNode)
{
    using NanoNodeT = typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type;
    openvdb::Coord &ijk = const_cast<openvdb::Coord&>(openNode->origin());
    NanoNodeT *nanoNode = PtrAdd<NanoNodeT>(mBufferPtr, reinterpret_cast<int64_t&>(ijk));
    Coord tmp = nanoNode->origin();
    ijk[0] = tmp[0];
    ijk[1] = tmp[1];
    ijk[2] = tmp[2];
    return nanoNode;
}// OpenToNanoVDB::decode

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename NodeT>
struct OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::NodePair {
    using OpenNodeT = NodeT;
    using NanoNodeT = typename NanoNode<NanoBuildT, OpenNodeT::LEVEL>::Type;
    NodePair(const NodeT *ptr, size_t n) : node(ptr), offset(n) {}
    const NodeT *node;// pointer to OpenVDB node
    uint64_t     offset;// byte offset to matching NanoVDB node, relative to the first
};// OpenToNanoVDB::NodePair

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
struct OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::BlindMetaData
{
    BlindMetaData(const std::string& n, const std::string& t, size_t i, size_t c, size_t s)
        : name(n)
        , typeName(t)
        , index(i)
        , count(c)
        , size(AlignUp<NANOVDB_DATA_ALIGNMENT>(c * s))
    {
    }
    const std::string name, typeName;
    const size_t      index, count, size;
    bool              operator<(const BlindMetaData& other) const { return index < other.index; } // required by std::set
}; // OpenToNanoVDB::BlindMetaData

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename T>
inline typename std::enable_if<!std::is_same<T, openvdb::tools::PointIndexGrid>::value &&
                               !std::is_same<T, openvdb::points::PointDataGrid>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::preProcessMetadata(const T& openGrid)
{
    mBlindMetaData.clear();
    const size_t length = openGrid.getName().length();
    if (length >= GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", 0, 1, length + 1);// Null-terminated byte strings
    }
}// OpenToNanoVDB::preProcessMetadata<T>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename T>
inline typename std::enable_if<std::is_same<T, openvdb::tools::PointIndexGrid>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::preProcessMetadata(const T& openGrid)
{
    mBlindMetaData.clear();
    if (const uint64_t pointCount = this->pointCount()) {
        mBlindMetaData.emplace("index", "uint32_t", 0, pointCount, sizeof(uint32_t));
    }
    const size_t length = openGrid.getName().length();
    if (length >= GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", mBlindMetaData.size(), 1, length + 1);// Null-terminated byte strings
    }
}// OpenToNanoVDB::preProcessMetadata<PointIndexGrid>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template <typename T>
inline typename std::enable_if<std::is_same<T, openvdb::points::PointDataGrid>::value>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::preProcessMetadata(const T& openGrid)
{
    mBlindMetaData.clear();
    size_t counter = 0;
    if (const uint64_t pointCount = this->pointCount()) {
        auto *openLeaf = openGrid.tree().cbeginLeaf().getLeaf();
        const auto& attributeSet = openLeaf->attributeSet();
        const auto& descriptor = attributeSet.descriptor();
        const auto& nameMap = descriptor.map();
        for (auto it = nameMap.begin(); it != nameMap.end(); ++it) {
            const size_t index = it->second;
            auto&        attArray = openLeaf->constAttributeArray(index);
            mBlindMetaData.emplace(it->first, descriptor.valueType(index), index, pointCount, attArray.valueTypeSize());
        }
        counter += nameMap.size();
    }
    const size_t length = openGrid.getName().length();
    if (length >= GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", counter, 1, length + 1);// Null-terminated byte strings
    }
}// OpenToNanoVDB::preProcessMetadata<PointDataGrid>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<!std::is_same<T, openvdb::tools::PointIndexGrid>::value &&
                               !std::is_same<T, openvdb::points::PointDataGrid>::value,GridBlindMetaData*>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    processMetadata(const T& openGrid)
{
    if (mBlindMetaData.empty()) {
        return nullptr;
    }
    assert(mBlindMetaData.size() == 1);// only the grid name is expected
    auto it = mBlindMetaData.cbegin();
    assert(it->name == "grid name" && it->typeName == "uint8_t" && it->index == 0);
    assert(openGrid.getName().length() >= GridData::MaxNameSize);
    auto *metaData = reinterpret_cast<GridBlindMetaData*>(mBufferPtr + mBufferOffsets[6]);
    auto *blindData = reinterpret_cast<char*>(mBufferPtr + mBufferOffsets[7]);
    // write the blind meta data
    metaData->setBlindData(blindData);
    metaData->mElementCount = it->count;
    metaData->mFlags = 0;
    metaData->mSemantic  = GridBlindDataSemantic::Unknown;
    metaData->mDataClass = GridBlindDataClass::GridName;
    metaData->mDataType  = GridType::Unknown;
    // write the actual bind data
    strcpy(blindData, openGrid.getName().c_str());
    return metaData;
}// OpenToNanoVDB::processMetadata<T>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<T, openvdb::tools::PointIndexGrid>::value,GridBlindMetaData*>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processMetadata(const T& openGrid)
{
    if (mBlindMetaData.empty()) {
        return nullptr;
    }
    assert(mBlindMetaData.size() == 1 || mBlindMetaData.size() == 2);// point index and maybe long grid name
    auto *metaData = reinterpret_cast<GridBlindMetaData*>(mBufferPtr + mBufferOffsets[6]);
    auto *blindData = reinterpret_cast<char*>(mBufferPtr + mBufferOffsets[7]);

    auto it = mBlindMetaData.cbegin();
    const uint32_t leafCount = static_cast<uint32_t>(mArray0.size());

    using LeafDataT = typename NanoLeafT::DataType;
    uint8_t* ptr = mBufferPtr + mBufferOffsets[5];

    auto *data0 = reinterpret_cast<LeafDataT*>(ptr + mArray0[0].offset);
    data0->mMinimum = 0; // start of prefix sum
    data0->mMaximum = data0->mValues[NanoLeafT::SIZE - 1u];
    for (uint32_t i = 1; i < leafCount; ++i) {
        auto *data1 = reinterpret_cast<LeafDataT*>(ptr + mArray0[i].offset);
        data1->mMinimum = data0->mMinimum + data0->mMaximum;
        data1->mMaximum = data1->mValues[NanoLeafT::SIZE - 1u];
        data0 = data1;
    }

    // write blind meta data for the point offsets
    assert(it->count == data0->mMinimum + data0->mMaximum);
    assert(it->name == "index" && it->typeName == "uint32_t" && it->index == 0);
    metaData[0].setBlindData( blindData );
    metaData[0].mElementCount = it->count;
    metaData[0].mFlags = 0;
    metaData[0].mSemantic = GridBlindDataSemantic::Unknown;
    metaData[0].mDataClass = GridBlindDataClass::IndexArray;
    metaData[0].mDataType = GridType::UInt32;
    if (it->name.length() >= GridBlindMetaData::MaxNameSize) {
        std::stringstream ss;
        ss << "Point attribute name \"" << it->name << "\" is more than " << (GridBlindMetaData::MaxNameSize-1) << " characters";
        OPENVDB_THROW(openvdb::ValueError, ss.str());
    }
    memcpy(metaData[0].mName, it->name.c_str(), it->name.size() + 1);

    // write point offsets as blind data
    forEach(mArray0, 16, [&](const auto& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                auto *data = reinterpret_cast<LeafDataT*>(ptr + mArray0[i].offset);
                uint32_t* p = reinterpret_cast<uint32_t*>(blindData) + data->mMinimum;
                for (uint32_t idx : mArray0[i].node->indices()) *p++ = idx;
            }
    });
    blindData += it->size;// add point offsets

    // write long grid name if it exists
    ++it;
    if (it != mBlindMetaData.end()) {
        assert(it->name == "grid name" && it->typeName == "uint8_t" && it->index == 1);
        assert(openGrid.getName().length() >= GridData::MaxNameSize);
        metaData[1].setBlindData( blindData );
        metaData[1].mElementCount = it->count;
        metaData[1].mFlags = 0;
        metaData[1].mSemantic = GridBlindDataSemantic::Unknown;
        metaData[1].mDataClass = GridBlindDataClass::GridName;
        metaData[1].mDataType = GridType::Unknown;
        strcpy(blindData, openGrid.getName().c_str());
    }
    return metaData;
}// OpenToNanoVDB::processMetadata<PointIndex32>

//================================================================================================

template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename T>
inline typename std::enable_if<std::is_same<T, openvdb::points::PointDataGrid>::value,GridBlindMetaData*>::type
OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::processMetadata(const T& openGrid)
{
    if (mBlindMetaData.empty()) {
        return nullptr;
    }

    auto *metaData = reinterpret_cast<GridBlindMetaData*>(mBufferPtr + mBufferOffsets[6]);
    auto *blindData = reinterpret_cast<char*>(mBufferPtr + mBufferOffsets[7]);

    const uint32_t leafCount = static_cast<uint32_t>(mArray0.size());

    using LeafDataT = typename NanoLeafT::DataType;
    uint8_t* ptr = mBufferPtr + mBufferOffsets[5];

    auto *data0 = reinterpret_cast<LeafDataT*>(ptr + mArray0[0].offset);
    data0->mMinimum = 0; // start of prefix sum
    data0->mMaximum = data0->mValues[NanoLeafT::SIZE - 1u];
    for (uint32_t i = 1; i < leafCount; ++i) {
        auto *data1 = reinterpret_cast<LeafDataT*>(ptr + mArray0[i].offset);
        data1->mMinimum = data0->mMinimum + data0->mMaximum;
        data1->mMaximum = data1->mValues[NanoLeafT::SIZE - 1u];
        data0 = data1;
    }

    size_t i=0;
    for (auto it = mBlindMetaData.cbegin(); it != mBlindMetaData.end(); ++it, ++i) {
        metaData[i].setBlindData( blindData );
        metaData[i].mElementCount = it->count;
        metaData[i].mFlags = 0;
        if (it->name == "grid name") {
            metaData[i].mSemantic = GridBlindDataSemantic::Unknown;
            metaData[i].mDataClass = GridBlindDataClass::GridName;
            metaData[i].mDataType = GridType::Unknown;
            assert(openGrid.getName().length() >= GridData::MaxNameSize);
            strcpy((char*)blindData, openGrid.getName().c_str());
        } else {
            assert(it->count == data0->mMinimum + data0->mMaximum);
            metaData[i].mDataClass = GridBlindDataClass::AttributeArray;
            if (it->name.length()>= GridBlindMetaData::MaxNameSize) {
                std::stringstream ss;
                ss << "Point attribute name \"" << it->name << "\" is more than " << (GridBlindMetaData::MaxNameSize-1) << " characters";
                OPENVDB_THROW(openvdb::ValueError, ss.str());
            }

            memcpy(metaData[i].mName, it->name.c_str(), it->name.size() + 1);
            if (it->typeName == "vec3s") {
                metaData[i].mDataType = GridType::Vec3f;
                this->copyPointAttribute(it->index, (openvdb::Vec3f*)blindData);
                if (it->name == "P") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointPosition;
                } else if (it->name == "V") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointVelocity;
                } else if (it->name == "Cd") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointColor;
                } else if (it->name == "N") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointNormal;
                } else {
                    metaData[i].mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (it->typeName == "int32") {
                metaData[i].mDataType = GridType::Int32;
                this->copyPointAttribute(it->index, (int32_t*)blindData);
                if (it->name == "id") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointId;
                } else {
                    metaData[i].mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (it->typeName == "int64") {
                metaData[i].mDataType = GridType::Int64;
                this->copyPointAttribute(it->index, (int64_t*)blindData);
                if (it->name == "id") {
                    metaData[i].mSemantic = GridBlindDataSemantic::PointId;
                } else {
                    metaData[i].mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (it->typeName == "float") {
                metaData[i].mDataType = GridType::Float;
                metaData[i].mSemantic = GridBlindDataSemantic::Unknown;
                this->copyPointAttribute(it->index, (float*)blindData);
            } else {
                std::stringstream ss;
                ss << "Unsupported point attribute type: \"" << it->typeName << "\"";
                OPENVDB_THROW(openvdb::ValueError, ss.str());
            }
        }
        blindData += it->size;
    } // loop over bind data
    return metaData;
}// OpenToNanoVDB::processMetadata<PointDataIndex32>

//================================================================================================


template<typename OpenBuildT, typename NanoBuildT, typename OracleT, typename BufferT>
template<typename AttT, typename CodecT>
inline void OpenToNanoVDB<OpenBuildT,  NanoBuildT,  OracleT, BufferT>::
    copyPointAttribute(size_t attIdx, AttT *attPtr)
{
    static_assert(std::is_same<typename OpenLeafT::ValueType, openvdb::PointDataIndex32>::value, "Expected value to openvdb::PointData");
    using LeafDataT = typename NanoLeafT::DataType;
    using HandleT = openvdb::points::AttributeHandle<AttT, CodecT>;
    forEach(mArray0, 16, [&](const auto& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto* openLeaf = mArray0[i].node;
            auto *nanoData = reinterpret_cast<LeafDataT*>(ptr + mArray0[i].offset);
            HandleT handle(openLeaf->constAttributeArray(attIdx));
            AttT* p = attPtr + nanoData->mMinimum;
            for (auto iter = openLeaf->beginIndexOn(); iter; ++iter) {
                *p++ = handle.get(*iter);
            }
        }
    });
}// OpenToNanoVDB::copyPointAttribute

//================================================================================================

template<typename BufferT, typename OpenTreeT, typename NanoBuildT>
GridHandle<BufferT>
openToNanoVDB(const openvdb::Grid<OpenTreeT>& grid,
              StatsMode       sMode,
              ChecksumMode    cMode,
              int             verbose)
{
    using OpenBuildT = typename OpenTreeT::BuildType;
    OpenToNanoVDB<OpenBuildT, NanoBuildT, AbsDiff, BufferT> s;
    return s(grid, sMode, cMode, verbose);
}// openToNanoVDB

//================================================================================================

template<typename BufferT>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base,
              StatsMode                     sMode,
              ChecksumMode                  cMode,
              int                           verbose)
{
    // We need to define these types because they are not defined in OpenVDB
    using openvdb_Vec4fTree = typename openvdb::tree::Tree4<openvdb::Vec4f, 5, 4, 3>::Type;
    using openvdb_Vec4dTree = typename openvdb::tree::Tree4<openvdb::Vec4d, 5, 4, 3>::Type;
    using openvdb_Vec4fGrid = openvdb::Grid<openvdb_Vec4fTree>;
    using openvdb_Vec4dGrid = openvdb::Grid<openvdb_Vec4dTree>;

    if (auto grid = openvdb::GridBase::grid<openvdb::FloatGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::FloatTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::DoubleGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::DoubleTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int32Grid>(base)) {
        return openToNanoVDB<BufferT, openvdb::Int32Tree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int64Grid>(base)) {
        return openToNanoVDB<BufferT, openvdb::Int64Tree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Grid<openvdb::UInt32Tree>>(base)) {
        return openToNanoVDB<BufferT, openvdb::UInt32Tree>(*grid, sMode, cMode,  verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3fGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::Vec3fTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3dGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::Vec3dTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::tools::PointIndexGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::tools::PointIndexTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::points::PointDataGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::points::PointDataTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::MaskGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::MaskTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::BoolGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::BoolTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb_Vec4fGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb_Vec4fTree>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb_Vec4dGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb_Vec4dTree>(*grid, sMode, cMode, verbose);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unrecognized OpenVDB grid type");
    }
}// openToNanoVDB

} // namespace nanovdb

#endif // NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
