// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CreateNanoGrid.h

    \author Ken Museth

    \date June 26, 2020

    \note In the examples below we assume that @c srcGrid is a exiting grid of type
          SrcGridT = @c openvdb::FloatGrid, @c openvdb::FloatGrid or @c nanovdb::build::FloatGrid.

    \brief Convert any grid to a nanovdb grid of the same type, e.g. float->float
    \code
    auto handle = nanovdb::createNanoGrid(srcGrid);
    auto *dstGrid = handle.grid<float>();
    \endcode

    \brief Convert a grid to a nanovdb grid of a different type, e.g. float->half
    \code
    auto handle = nanovdb::createNanoGrid<SrcGridT,nanovdb::Fp16>(srcGrid);
    auto *dstGrid = handle.grid<nanovdb::Fp16>();
    \endcode

    \brief Convert a grid to a nanovdb grid of the same type but using a CUDA buffer
    \code
    auto handle = nanovdb::createNanoGrid<SrcGridT, float, nanovdb::CudaDeviceBuffer>(srcGrid);
    auto *dstGrid = handle.grid<float>();
    \endcode

    \brief Create a nanovdb grid that indices values in an existing source grid of any type.
           If DstBuildT = nanovdb::ValueIndex both active and in-active values are indexed
           and if DstBuildT = nanovdb::ValueOnIndex only active values are indexed.
    \code
    using DstBuildT = nanovdb::ValueIndex;// index both active an inactive values
    auto handle = nanovdb::createNanoGridSrcGridT,DstBuildT>(srcGrid,0,false,false);//no blind data, tile values or stats
    auto *dstGrid = handle.grid<DstBuildT>();
    \endcode

    \brief Create a NanoVDB grid from scratch
    \code
#if defined(NANOVDB_USE_OPENVDB)
    using SrcGridT = openvdb::FloatGrid;
#else
    using SrcGridT = nanovdb::build::FloatGrid;
#endif
    SrcGridT srcGrid(0.0f);// create an empty source grid
    auto srcAcc = srcGrid.getAccessor();// create an accessor
    srcAcc.setValue(nanovdb::Coord(1,2,3), 1.0f);// set a voxel value

    auto handle = nanovdb::createNanoGrid(srcGrid);// convert source grid to a grid handle
    auto dstGrid = handle.grid<float>();// get a pointer to the destination grid
    \endcode

    \brief Convert a base-pointer to an openvdb grid, denoted srcGrid, to a  nanovdb
           grid of the same type, e.g. float -> float or openvdb::Vec3f -> nanovdb::Vec3f
    \code
    auto handle = nanovdb::openToNanoVDB(*srcGrid);// convert source grid to a grid handle
    auto dstGrid = handle.grid<float>();// get a pointer to the destination grid
    \endcode

    \brief Converts any existing grid to a NanoVDB grid, for example:
           nanovdb::build::Grid<SrcBuildT> -> nanovdb::Grid<DstBuildT>
           nanovdb::Grid<SrcBuildT> -> nanovdb::Grid<DstBuildT>
           nanovdb::Grid<SrcBuildT> -> nanovdb::Grid<ValueIndex or ValueOnIndex>
           openvdb::Grid<SrcBuildT> -> nanovdb::Grid<DstBuildT>
           openvdb::Grid<PointIndex> -> nanovdb::Grid<PointIndex>
           openvdb::Grid<PointData> -> nanovdb::Grid<PointData>
           openvdb::Grid<SrcBuildT> -> nanovdb::Grid<ValueIndex or ValueOnIndex>

    \note This files replaces GridBuilder.h, IndexGridBuilder.h and OpenToNanoVDB.h
*/

#ifndef NANOVDB_CREATE_NANOGRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_CREATE_NANOGRID_H_HAS_BEEN_INCLUDED

#if defined(NANOVDB_USE_OPENVDB)
#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/tools/PointIndexGrid.h>
#endif

#include "GridBuilder.h"
#include "NodeManager.h"
#include "GridHandle.h"
#include "GridStats.h"
#include "GridChecksum.h"
#include "Range.h"
#include "Invoke.h"
#include "ForEach.h"
#include "Reduce.h"
#include "PrefixSum.h"
#include "DitherLUT.h"// for nanovdb::DitherLUT

#include <limits>
#include <sstream> // for stringstream
#include <vector>
#include <set>
#include <cstring> // for memcpy
#include <type_traits>

namespace nanovdb {

// Forward declarations (defined below)
template <typename> class CreateNanoGrid;
class AbsDiff;
template <typename> struct MapToNano;

//================================================================================================

#if defined(NANOVDB_USE_OPENVDB)
/// @brief Forward declaration of free-standing function that converts an OpenVDB GridBase into a NanoVDB GridHandle
/// @tparam BufferT Type of the buffer used to allocate the destination grid
/// @param base Shared pointer to a base openvdb grid to be converted
/// @param sMode Mode for computing statistics of the destination grid
/// @param cMode Mode for computing checksums of the destination grid
/// @param verbose Mode of verbosity
/// @return Handle to the destination NanoGrid
template<typename BufferT = HostBuffer>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base,
              StatsMode                     sMode = StatsMode::Default,
              ChecksumMode                  cMode = ChecksumMode::Default,
              int                           verbose = 0);
#endif

//================================================================================================

/// @brief Freestanding function that creates a NanoGrid<T> from any source grid
/// @tparam SrcGridT Type of in input (source) grid, e.g. openvdb::Grid or nanovdb::Grid
/// @tparam DstBuildT Type of values in the output (destination) nanovdb Grid, e.g. float or nanovdb::Fp16
/// @tparam BufferT Type of the buffer used ti allocate the destination grid
/// @param srcGrid Input (source) grid to be converted
/// @param sMode  Mode for computing statistics of the destination grid
/// @param cMode  Mode for computing checksums of the destination grid
/// @param verbose Mode of verbosity
/// @param buffer Instance of a buffer used for allocation
/// @return Handle to the destination NanoGrid
template<typename SrcGridT,
         typename DstBuildT = typename MapToNano<typename SrcGridT::BuildType>::type,
         typename BufferT = HostBuffer>
typename disable_if<BuildTraits<DstBuildT>::is_index || BuildTraits<DstBuildT>::is_Fp, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode = StatsMode::Default,
               ChecksumMode cMode = ChecksumMode::Default,
               int verbose = 0,
               const BufferT &buffer = BufferT());

//================================================================================================

/// @brief Freestanding function that creates a NanoGrid<ValueIndex> or NanoGrid<ValueOnIndex> from any source grid
/// @tparam SrcGridT Type of in input (source) grid, e.g. openvdb::Grid or nanovdb::Grid
/// @tparam DstBuildT If ValueIndex all (active and inactive) values are indexed and if
///         it is ValueOnIndex only active values are indexed.
/// @tparam BufferT BufferT Type of the buffer used ti allocate the destination grid
/// @param channels If non-zero the values (active or all) in @c srcGrid are encoded as blind
///                 data in the output index grid. @c channels indicates the number of copies
///                 of these blind data
/// @param includeStats If true all tree nodes will includes indices for stats, i.e. min/max/avg/std-div
/// @param includeTiles If false on values in leaf nodes are indexed
/// @param verbose Mode of verbosity
/// @param buffer Instance of a buffer used for allocation
/// @return Handle to the destination NanoGrid<T> where T = ValueIndex or ValueOnIndex
template<typename SrcGridT,
         typename DstBuildT = typename MapToNano<typename SrcGridT::BuildType>::type,
         typename BufferT = HostBuffer>
typename enable_if<BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               uint32_t channels = 0u,
               bool includeStats = true,
               bool includeTiles = true,
               int verbose = 0,
               const BufferT &buffer = BufferT());

//================================================================================================

/// @brief Freestanding function to create a NanoGrid<FpN> from any source grid
/// @tparam SrcGridT Type of in input (source) grid, e.g. openvdb::Grid or nanovdb::Grid
/// @tparam DstBuildT = FpN, i.e. variable bit-width of the output grid
/// @tparam OracleT Type of the oracle used to determine the local bit-width, i.e. N in FpN
/// @tparam BufferT Type of the buffer used to allocate the destination grid
/// @param srcGrid Input (source) grid to be converted
/// @param ditherOn switch to enable or disable dithering of quantization error
/// @param sMode Mode for computing statistics of the destination grid
/// @param cMode Mode for computing checksums of the destination grid
/// @param verbose Mode of verbosity
/// @param oracle Instance of a oracle used  to determine the local bit-width, i.e. N in FpN
/// @param buffer Instance of a buffer used for allocation
/// @return Handle to the destination NanoGrid
template<typename SrcGridT,
         typename DstBuildT = typename MapToNano<typename SrcGridT::BuildType>::type,
         typename OracleT = AbsDiff,
         typename BufferT = HostBuffer>
typename enable_if<is_same<FpN, DstBuildT>::value, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode = StatsMode::Default,
               ChecksumMode cMode = ChecksumMode::Default,
               bool ditherOn = false,
               int verbose = 0,
               const OracleT &oracle = OracleT(),
               const BufferT &buffer = BufferT());

//================================================================================================

/// @brief Freestanding function to create a NanoGrid<FpX> from any source grid, X=4,8,16
/// @tparam SrcGridT Type of in input (source) grid, e.g. openvdb::Grid or nanovdb::Grid
/// @tparam DstBuildT = Fp4, Fp8 or Fp16, i.e. quantization bit-width of the output grid
/// @tparam BufferT Type of the buffer used to allocate the destination grid
/// @param srcGrid Input (source) grid to be converted
/// @param ditherOn switch to enable or disable dithering of quantization error
/// @param sMode Mode for computing statistics of the destination grid
/// @param cMode Mode for computing checksums of the destination grid
/// @param verbose Mode of verbosity
/// @param buffer Instance of a buffer used for allocation
/// @return Handle to the destination NanoGrid
template<typename SrcGridT,
         typename DstBuildT = typename MapToNano<typename SrcGridT::BuildType>::type,
         typename BufferT = HostBuffer>
typename enable_if<BuildTraits<DstBuildT>::is_FpX, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode = StatsMode::Default,
               ChecksumMode cMode = ChecksumMode::Default,
               bool ditherOn = false,
               int verbose = 0,
               const BufferT &buffer = BufferT());

//================================================================================================

/// @brief Compression oracle based on absolute difference
class AbsDiff
{
    float mTolerance;// absolute error tolerance
public:
    /// @note The default value of -1 means it's un-initialized!
    AbsDiff(float tolerance = -1.0f) : mTolerance(tolerance) {}
    AbsDiff(const AbsDiff&) = default;
    ~AbsDiff() = default;
    operator bool() const {return mTolerance>=0.0f;}
    void init(nanovdb::GridClass gClass, float background) {
        if (gClass == GridClass::LevelSet) {
            static const float halfWidth = 3.0f;
            mTolerance = 0.1f * background / halfWidth;// range of ls: [-3dx; 3dx]
        } else if (gClass == GridClass::FogVolume) {
            mTolerance = 0.01f;// range of FOG volumes: [0;1]
        } else {
            mTolerance = 0.0f;
        }
    }
    void  setTolerance(float tolerance) { mTolerance = tolerance; }
    float getTolerance() const { return mTolerance; }
    /// @brief Return true if the approximate value is within the accepted
    ///        absolute error bounds of the exact value.
    ///
    /// @details Required member method
    bool  operator()(float exact, float approx) const
    {
        return Abs(exact - approx) <= mTolerance;
    }
};// AbsDiff

inline std::ostream& operator<<(std::ostream& os, const AbsDiff& diff)
{
    os << "Absolute tolerance: " << diff.getTolerance();
    return os;
}

//================================================================================================

/// @brief Compression oracle based on relative difference
class RelDiff
{
    float mTolerance;// relative error tolerance
public:
    /// @note The default value of -1 means it's un-initialized!
    RelDiff(float tolerance = -1.0f) : mTolerance(tolerance) {}
    RelDiff(const RelDiff&) = default;
    ~RelDiff() = default;
    operator bool() const {return mTolerance>=0.0f;}
    void  setTolerance(float tolerance) { mTolerance = tolerance; }
    float getTolerance() const { return mTolerance; }
    /// @brief Return true if the approximate value is within the accepted
    ///        relative error bounds of the exact value.
    ///
    /// @details Required member method
    bool  operator()(float exact, float approx) const
    {
        return  Abs(exact - approx)/Max(Abs(exact), Abs(approx)) <= mTolerance;
    }
};// RelDiff

inline std::ostream& operator<<(std::ostream& os, const RelDiff& diff)
{
    os << "Relative tolerance: " << diff.getTolerance();
    return os;
}

//================================================================================================

/// @brief The NodeAccessor provides a uniform API for accessing nodes got NanoVDB, OpenVDB and build Grids
///
/// @note General implementation that works with nanovdb::build::Grid
template <typename GridT>
class NodeAccessor
{
public:
    static constexpr bool IS_OPENVDB = false;
    static constexpr bool IS_NANOVDB = false;
    using BuildType = typename GridT::BuildType;
    using ValueType = typename GridT::ValueType;
    using GridType = GridT;
    using TreeType = typename GridT::TreeType;
    using RootType = typename TreeType::RootNodeType;
    template<int LEVEL>
    using NodeType = typename NodeTrait<const TreeType, LEVEL>::type;
    NodeAccessor(const GridT &grid) : mMgr(const_cast<GridT&>(grid)) {}
    const GridType& grid() const {return mMgr.grid();}
    const TreeType& tree() const {return mMgr.tree();}
    const RootType& root() const {return mMgr.root();}
    uint64_t nodeCount(int level) const { return mMgr.nodeCount(level); }
    template <int LEVEL>
    const NodeType<LEVEL>& node(uint32_t i) const {return mMgr.template node<LEVEL>(i); }
    const std::string& getName() const {return this->grid().getName();};
    bool hasLongGridName() const {return this->grid().getName().length() >= GridData::MaxNameSize;}
    const nanovdb::Map& map() const {return this->grid().map();}
    GridClass gridClass() const {return this->grid().gridClass();}
private:
    build::NodeManager<GridT> mMgr;
};// NodeAccessor<GridT>

//================================================================================================

/// @brief Template specialization for nanovdb::Grid which is special since its NodeManage
///         uses a handle in order to support node access on the GPU!
template <typename BuildT>
class NodeAccessor< NanoGrid<BuildT> >
{
public:
    static constexpr bool IS_OPENVDB = false;
    static constexpr bool IS_NANOVDB = true;
    using BuildType = BuildT;
    using BufferType = HostBuffer;
    using GridType = NanoGrid<BuildT>;
    using ValueType = typename GridType::ValueType;
    using TreeType = typename GridType::TreeType;
    using RootType = typename TreeType::RootType;
    template<int LEVEL>
    using NodeType = typename NodeTrait<TreeType, LEVEL>::type;
    NodeAccessor(const GridType &grid)
        : mHandle(createNodeManager<BuildT, BufferType>(grid))
        , mMgr(*(mHandle.template mgr<BuildT>())) {}
    const GridType& grid() const {return mMgr.grid();}
    const TreeType& tree() const {return mMgr.tree();}
    const RootType& root() const {return mMgr.root();}
    uint64_t nodeCount(int level) const { return mMgr.nodeCount(level); }
    template <int LEVEL>
    const NodeType<LEVEL>& node(uint32_t i) const {return mMgr.template node<LEVEL>(i); }
    std::string getName() const {return std::string(this->grid().gridName());};
    bool hasLongGridName() const {return this->grid().hasLongGridName();}
    const nanovdb::Map& map() const {return this->grid().map();}
    GridClass gridClass() const {return this->grid().gridClass();}
private:
    NodeManagerHandle<BufferType> mHandle;
    const NodeManager<BuildT>    &mMgr;
};// NodeAccessor<nanovdb::Grid>

//================================================================================================

/// @brief Trait that maps any type to the corresponding nanovdb type
/// @tparam T Type to be mapped
template<typename T>
struct MapToNano { using type = T; };

#if defined(NANOVDB_USE_OPENVDB)

template<>
struct MapToNano<openvdb::ValueMask> {using type = nanovdb::ValueMask;};
template<typename T>
struct MapToNano<openvdb::math::Vec3<T>>{using type = nanovdb::Vec3<T>;};
template<typename T>
struct MapToNano<openvdb::math::Vec4<T>>{using type = nanovdb::Vec4<T>;};
template<>
struct MapToNano<openvdb::PointIndex32> {using type = uint32_t;};
template<>
struct MapToNano<openvdb::PointDataIndex32> {using type = uint32_t;};

/// Templated Grid with default 32->16->8 configuration
template <typename BuildT>
using OpenLeaf = openvdb::tree::LeafNode<BuildT,3>;
template <typename BuildT>
using OpenLower = openvdb::tree::InternalNode<OpenLeaf<BuildT>,4>;
template <typename BuildT>
using OpenUpper = openvdb::tree::InternalNode<OpenLower<BuildT>,5>;
template <typename BuildT>
using OpenRoot = openvdb::tree::RootNode<OpenUpper<BuildT>>;
template <typename BuildT>
using OpenTree = openvdb::tree::Tree<OpenRoot<BuildT>>;
template <typename BuildT>
using OpenGrid = openvdb::Grid<OpenTree<BuildT>>;

//================================================================================================

/// @brief Template specialization for openvdb::Grid
template <typename BuildT>
class NodeAccessor<OpenGrid<BuildT>>
{
public:
    static constexpr bool IS_OPENVDB = true;
    static constexpr bool IS_NANOVDB = false;
    using BuildType = BuildT;
    using GridType = OpenGrid<BuildT>;
    using ValueType = typename GridType::ValueType;
    using TreeType = OpenTree<BuildT>;
    using RootType = OpenRoot<BuildT>;
    template<int LEVEL>
    using NodeType = typename NodeTrait<const TreeType, LEVEL>::type;
    NodeAccessor(const GridType &grid) : mMgr(const_cast<GridType&>(grid)) {
        const auto mat4 = this->grid().transform().baseMap()->getAffineMap()->getMat4();
        mMap.set(mat4, mat4.inverse());
    }
    const GridType& grid() const {return mMgr.grid();}
    const TreeType& tree() const {return mMgr.tree();}
    const RootType& root() const {return mMgr.root();}
    uint64_t nodeCount(int level) const { return mMgr.nodeCount(level); }
    template <int LEVEL>
    const NodeType<LEVEL>& node(uint32_t i) const {return mMgr.template node<LEVEL>(i); }
    std::string getName() const { return this->grid().getName(); };
    bool hasLongGridName() const {return this->grid().getName().length() >= GridData::MaxNameSize;}
    const nanovdb::Map& map() const {return mMap;}
    GridClass gridClass() const {
        switch (this->grid().getGridClass()) {
        case openvdb::GRID_LEVEL_SET:
            if (!is_floating_point<BuildT>::value) OPENVDB_THROW(openvdb::ValueError, "processGrid: Level sets are expected to be floating point types");
            return GridClass::LevelSet;
        case openvdb::GRID_FOG_VOLUME:
            return GridClass::FogVolume;
        case openvdb::GRID_STAGGERED:
            return GridClass::Staggered;
        default:
            return GridClass::Unknown;
        }
    }
private:
    build::NodeManager<GridType> mMgr;
    nanovdb::Map                 mMap;
};// NodeAccessor<openvdb::Grid<T>>

//================================================================================================

/// @brief Template specialization for openvdb::tools::PointIndexGrid
template <>
class NodeAccessor<openvdb::tools::PointIndexGrid>
{
public:
    static constexpr bool IS_OPENVDB = true;
    static constexpr bool IS_NANOVDB = false;
    using BuildType = openvdb::PointIndex32;
    using GridType = openvdb::tools::PointIndexGrid;
    using TreeType = openvdb::tools::PointIndexTree;
    using RootType = typename TreeType::RootNodeType;
    using ValueType = typename GridType::ValueType;
    template<int LEVEL>
    using NodeType = typename NodeTrait<const TreeType, LEVEL>::type;
    NodeAccessor(const GridType &grid) : mMgr(const_cast<GridType&>(grid)) {
        const auto mat4 = this->grid().transform().baseMap()->getAffineMap()->getMat4();
        mMap.set(mat4, mat4.inverse());
    }
    const GridType& grid() const {return mMgr.grid();}
    const TreeType& tree() const {return mMgr.tree();}
    const RootType& root() const {return mMgr.root();}
    uint64_t nodeCount(int level) const { return mMgr.nodeCount(level); }
    template <int LEVEL>
    const NodeType<LEVEL>& node(uint32_t i) const {return mMgr.template node<LEVEL>(i); }
    std::string getName() const { return this->grid().getName(); };
    bool hasLongGridName() const {return this->grid().getName().length() >= GridData::MaxNameSize;}
    const nanovdb::Map& map() const {return mMap;}
    GridClass gridClass() const {return GridClass::PointIndex;}
private:
    build::NodeManager<GridType> mMgr;
    nanovdb::Map                 mMap;
};// NodeAccessor<openvdb::tools::PointIndexGrid>

//================================================================================================

// @brief Template specialization for openvdb::points::PointDataGrid
template <>
class NodeAccessor<openvdb::points::PointDataGrid>
{
public:
    static constexpr bool IS_OPENVDB = true;
    static constexpr bool IS_NANOVDB = false;
    using BuildType = openvdb::PointDataIndex32;
    using GridType = openvdb::points::PointDataGrid;
    using TreeType = openvdb::points::PointDataTree;
    using RootType = typename TreeType::RootNodeType;
    using ValueType = typename GridType::ValueType;
    template<int LEVEL>
    using NodeType = typename NodeTrait<const TreeType, LEVEL>::type;
    NodeAccessor(const GridType &grid) : mMgr(const_cast<GridType&>(grid)) {
        const auto mat4 = this->grid().transform().baseMap()->getAffineMap()->getMat4();
        mMap.set(mat4, mat4.inverse());
    }
    const GridType& grid() const {return mMgr.grid();}
    const TreeType& tree() const {return mMgr.tree();}
    const RootType& root() const {return mMgr.root();}
    uint64_t nodeCount(int level) const { return mMgr.nodeCount(level); }
    template <int LEVEL>
    const NodeType<LEVEL>& node(uint32_t i) const {return mMgr.template node<LEVEL>(i); }
    std::string getName() const { return this->grid().getName(); };
    bool hasLongGridName() const {return this->grid().getName().length() >= GridData::MaxNameSize;}
    const nanovdb::Map& map() const {return mMap;}
    GridClass gridClass() const {return GridClass::PointData;}
private:
    build::NodeManager<GridType> mMgr;
    nanovdb::Map                 mMap;
};// NodeAccessor<openvdb::points::PointDataGrid>

#endif// NANOVDB_USE_OPENVDB

//================================================================================================

/// @brief Creates any nanovdb Grid from any source grid (certain combinations are obviously not allowed)
template <typename SrcGridT>
class CreateNanoGrid
{
public:
    // SrcGridT can be either openvdb::Grid, nanovdb::Grid or nanovdb::build::Grid
    using SrcNodeAccT = NodeAccessor<SrcGridT>;
    using SrcBuildT = typename SrcNodeAccT::BuildType;
    using SrcValueT = typename SrcNodeAccT::ValueType;
    using SrcTreeT  = typename SrcNodeAccT::TreeType;
    using SrcRootT  = typename SrcNodeAccT::RootType;
    template <int LEVEL>
    using SrcNodeT = typename NodeTrait<SrcRootT, LEVEL>::type;

    /// @brief Constructor from a source grid
    /// @param srcGrid Source grid of type SrcGridT
    CreateNanoGrid(const SrcGridT &srcGrid);

    /// @brief Constructor from a source node accessor (defined above)
    /// @param srcNodeAcc Source node accessor of type SrcNodeAccT
    CreateNanoGrid(const SrcNodeAccT &srcNodeAcc);

    /// @brief Set the level of verbosity
    /// @param mode level of verbosity, mode=0 means quiet
    void setVerbose(int mode = 1) { mVerbose = mode; }

    /// @brief Enable or disable dithering, i.e. randomization of the quantization error.
    /// @param on enable or disable dithering
    /// @warning Dithering only has an affect when DstBuildT = {Fp4, Fp8, Fp16, FpN}
    void enableDithering(bool on = true) { mDitherOn = on; }

    /// @brief Set the mode used for computing statistics of the destination grid
    /// @param mode specify the mode of statistics
    void setStats(StatsMode mode = StatsMode::Default) { mStats = mode; }

    /// @brief Set the mode used for computing checksums of the destination grid
    /// @param mode specify the mode of checksum
    void setChecksum(ChecksumMode mode = ChecksumMode::Default) { mChecksum = mode; }

    /// @brief Converts the source grid into a nanovdb grid with the specified destination build type
    /// @tparam DstBuildT build type of the destination, output, grid
    /// @tparam BufferT Type of the buffer used for allocating the destination grid
    /// @param buffer instance of the buffer use for allocation
    /// @return Return an instance of a GridHandle (invoking move semantics)
    /// @note This version is when DstBuildT != {FpN, ValueIndex, ValueOnIndex}
    template<typename DstBuildT = typename MapToNano<SrcBuildT>::type, typename BufferT = HostBuffer>
    typename disable_if<is_same<DstBuildT, FpN>::value ||
                        BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
    getHandle(const BufferT &buffer = BufferT());

    /// @brief Converts the source grid into a nanovdb grid with variable bit quantization
    /// @tparam DstBuildT FpN, i.e. the destination grid uses variable bit quantization
    /// @tparam OracleT Type of oracle used to determine the N in FpN
    /// @tparam BufferT Type of the buffer used for allocating the destination grid
    /// @param oracle Instance of the oracle used to determine the N in FpN
    /// @param buffer instance of the buffer use for allocation
    /// @return Return an instance of a GridHandle (invoking move semantics)
    /// @note This version assumes DstBuildT == FpN
    template<typename DstBuildT = typename MapToNano<SrcBuildT>::type, typename OracleT = AbsDiff, typename BufferT = HostBuffer>
    typename enable_if<is_same<DstBuildT, FpN>::value, GridHandle<BufferT>>::type
    getHandle(const OracleT &oracle = OracleT(),
              const BufferT &buffer = BufferT());

    /// @brief Converts the source grid into a nanovdb grid with indices to external arrays of values
    /// @tparam DstBuildT ValueIndex or ValueOnIndex, i.e. index all or just active values
    /// @tparam BufferT Type of the buffer used for allocating the destination grid
    /// @param channels Number of copies of values encoded as blind data in the destination grid
    /// @param includeStats Specify if statics should be indexed
    /// @param includeTiles Specify if tile values, i.e. non-leaf-node-values, should be indexed
    /// @param buffer instance of the buffer use for allocation
    /// @return Return an instance of a GridHandle (invoking move semantics)
    template<typename DstBuildT = typename MapToNano<SrcBuildT>::type, typename BufferT = HostBuffer>
    typename enable_if<BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
    getHandle(uint32_t channels = 0u,
              bool includeStats = true,
              bool includeTiles = true,
              const BufferT &buffer = BufferT());

    /// @brief Add blind data to the destination grid
    /// @param name String name of the blind data
    /// @param dataSemantic Semantics of the blind data
    /// @param dataClass Class of the blind data
    /// @param dataType Type of the blind data
    /// @param count Element count of the blind data
    /// @param size Size of each element of the blind data
    /// @return Return the index used to access the blind data
    uint64_t addBlindData(const std::string& name,
                          GridBlindDataSemantic dataSemantic,
                          GridBlindDataClass dataClass,
                          GridType dataType,
                          size_t count, size_t size)
    {
        const size_t order = mBlindMetaData.size();
        mBlindMetaData.emplace(name, dataSemantic, dataClass, dataType, order, count, size);
        return order;
    }

    /// @brief This method only has affect when getHandle was called with DstBuildT = ValueIndex or ValueOnIndex
    /// @return Return the number of indexed values. If called before getHandle was called with
    ///         DstBuildT = ValueIndex or ValueOnIndex the return value is zero. Else it is a value larger than zero.
    uint64_t valueCount() const {return mValIdx[0].empty() ? 0u : mValIdx[0].back();}

    /// @brief Copy values from the source grid into a provided buffer
    /// @tparam DstBuildT Must be ValueIndex or ValueOnIndex, i.e. a index grid
    /// @param buffer point in which to write values
    template <typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    copyValues(SrcValueT *buffer);

private:

    // =========================================================

    template <typename T, int LEVEL>
    typename enable_if<!(is_same<T,FpN>::value&&LEVEL==0), typename NodeTrait<NanoRoot<T>, LEVEL>::type*>::type
    dstNode(uint64_t i) const {
        static_assert(LEVEL==0 || LEVEL==1 || LEVEL==2, "Expected LEVEL== {0,1,2}");
        using NodeT = typename NodeTrait<NanoRoot<T>, LEVEL>::type;
        return PtrAdd<NodeT>(mBufferPtr, mOffset[5-LEVEL]) + i;
    }
    template <typename T, int LEVEL>
    typename enable_if<is_same<T,FpN>::value && LEVEL==0, NanoLeaf<FpN>*>::type
    dstNode(uint64_t i) const {return PtrAdd<NanoLeaf<FpN>>(mBufferPtr, mCodec[i].offset);}

    template <typename T> NanoRoot<T>* dstRoot() const {return PtrAdd<NanoRoot<T>>(mBufferPtr, mOffset.root);}
    template <typename T> NanoTree<T>* dstTree() const {return PtrAdd<NanoTree<T>>(mBufferPtr, mOffset.tree);}
    template <typename T> NanoGrid<T>* dstGrid() const {return PtrAdd<NanoGrid<T>>(mBufferPtr, mOffset.grid);}
    GridBlindMetaData* dstMeta(uint32_t i) const { return PtrAdd<GridBlindMetaData>(mBufferPtr, mOffset.meta) + i;};

    // =========================================================

    template <typename DstBuildT>
    typename disable_if<is_same<FpN,DstBuildT>::value || BuildTraits<DstBuildT>::is_index>::type
    preProcess();

    template <typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    preProcess(uint32_t channels);

    template <typename DstBuildT, typename OracleT>
    typename enable_if<is_same<FpN, DstBuildT>::value>::type
    preProcess(OracleT oracle);

    // =========================================================

    // Below are private methods use to serialize nodes into NanoVDB
    template<typename DstBuildT, typename BufferT>
    GridHandle<BufferT> initHandle(const BufferT& buffer);

    // =========================================================

    template <typename DstBuildT>
    inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    postProcess(uint32_t channels);

    template <typename DstBuildT>
    inline typename disable_if<BuildTraits<DstBuildT>::is_index>::type
    postProcess();

    // ========================================================

    template<typename DstBuildT>
    typename disable_if<BuildTraits<DstBuildT>::is_special>::type
    processLeafs();

    template<typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    processLeafs();

    template<typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_FpX>::type
    processLeafs();

    template<typename DstBuildT>
    typename enable_if<is_same<FpN, DstBuildT>::value>::type
    processLeafs();

    template<typename DstBuildT>
    typename enable_if<is_same<bool, DstBuildT>::value>::type
    processLeafs();

    template<typename DstBuildT>
    typename enable_if<is_same<ValueMask, DstBuildT>::value>::type
    processLeafs();

    // =========================================================

    template<typename DstBuildT, int LEVEL>
    typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    processInternalNodes();

    template<typename DstBuildT, int LEVEL>
    typename enable_if<!BuildTraits<DstBuildT>::is_index>::type
    processInternalNodes();

    // =========================================================

    template <typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_index>::type
    processRoot();

    template <typename DstBuildT>
    typename enable_if<!BuildTraits<DstBuildT>::is_index>::type
    processRoot();

    // =========================================================

    template<typename DstBuildT>
    void processTree();

    template<typename DstBuildT>
    void processGrid();

    template <typename DstBuildT, int LEVEL>
    typename enable_if<BuildTraits<DstBuildT>::is_index, uint64_t>::type
    countTileValues(uint64_t valueCount);

    template <typename DstBuildT>
    typename enable_if<BuildTraits<DstBuildT>::is_index, uint64_t>::type
    countValues();

#if defined(NANOVDB_USE_OPENVDB)
    template<typename T = SrcGridT>
    typename disable_if<is_same<T, openvdb::tools::PointIndexGrid>::value ||
                               is_same<T, openvdb::points::PointDataGrid>::value, uint64_t>::type
    countPoints() const;

    template<typename T = SrcGridT>
    typename enable_if<is_same<T, openvdb::tools::PointIndexGrid>::value ||
                       is_same<T, openvdb::points::PointDataGrid>::value, uint64_t>::type
    countPoints() const;

    template<typename DstBuildT, typename AttT, typename CodecT = openvdb::points::UnknownCodec, typename T = SrcGridT>
    typename enable_if<is_same<openvdb::points::PointDataGrid, T>::value>::type
    copyPointAttribute(size_t attIdx, AttT *attPtr);
#else
    uint64_t countPoints() const {return 0u;}
#endif

    uint8_t*                 mBufferPtr;// pointer to the beginning of the destination nanovdb grid buffer
    struct BufferOffsets {
        uint64_t grid, tree, root, upper, lower, leaf, meta, blind, size;
        uint64_t operator[](int i) const { return *(reinterpret_cast<const uint64_t*>(this)+i); }
    }                        mOffset;
    int                      mVerbose;
    uint64_t                 mLeafNodeSize;// non-trivial when DstBuiltT = FpN

    std::unique_ptr<SrcNodeAccT> mSrcNodeAccPtr;// placeholder for potential local instance
    const SrcNodeAccT       &mSrcNodeAcc;
    struct BlindMetaData; // forward declaration
    std::set<BlindMetaData>  mBlindMetaData; // sorted according to BlindMetaData.order
    struct Codec { float min, max; uint64_t offset; uint8_t log2; };// used for adaptive bit-rate quantization
    std::unique_ptr<Codec[]> mCodec;// defines a codec per leaf node when DstBuildT = FpN
    StatsMode                mStats;
    ChecksumMode             mChecksum;
    bool                     mDitherOn, mIncludeStats, mIncludeTiles;
    std::vector<uint64_t>    mValIdx[3];// store id of first value in node
}; // CreateNanoGrid

//================================================================================================

template <typename SrcGridT>
CreateNanoGrid<SrcGridT>::CreateNanoGrid(const SrcGridT &srcGrid)
    : mVerbose(0)
    , mSrcNodeAccPtr(new SrcNodeAccT(srcGrid))
    , mSrcNodeAcc(*mSrcNodeAccPtr)
    , mStats(StatsMode::Default)
    , mChecksum(ChecksumMode::Default)
    , mDitherOn(false)
    , mIncludeStats(true)
    , mIncludeTiles(true)
{
}

//================================================================================================

template <typename SrcGridT>
CreateNanoGrid<SrcGridT>::CreateNanoGrid(const SrcNodeAccT &srcNodeAcc)
    : mVerbose(0)
    , mSrcNodeAccPtr(nullptr)
    , mSrcNodeAcc(srcNodeAcc)
    , mStats(StatsMode::Default)
    , mChecksum(ChecksumMode::Default)
    , mDitherOn(false)
    , mIncludeStats(true)
    , mIncludeTiles(true)
{
}

//================================================================================================

template <typename SrcGridT>
struct CreateNanoGrid<SrcGridT>::BlindMetaData
{
    BlindMetaData(const std::string& name,// name + used to derive GridBlindDataSemantic
                  const std::string& type,// used to derive GridType of blind data
                  GridBlindDataClass dataClass,
                  size_t i, size_t valueCount, size_t valueSize)
        : metaData(reinterpret_cast<GridBlindMetaData*>(new char[sizeof(GridBlindMetaData)]))
        , order(i)// sorted id of meta data
        , size(AlignUp<NANOVDB_DATA_ALIGNMENT>(valueCount * valueSize))
    {
        std::memset(metaData, 0, sizeof(GridBlindMetaData));// zero out all meta data
        if (name.length()>=GridData::MaxNameSize) throw std::runtime_error("blind data name exceeds limit");
        std::memcpy(metaData->mName, name.c_str(), name.length() + 1);
        metaData->mValueCount = valueCount;
        metaData->mSemantic = BlindMetaData::mapToSemantics(name);
        metaData->mDataClass = dataClass;
        metaData->mDataType = BlindMetaData::mapToType(type);
        metaData->mValueSize = valueSize;
        NANOVDB_ASSERT(metaData->isValid());
    }
    BlindMetaData(const std::string& name,// only name
                  GridBlindDataSemantic dataSemantic,
                  GridBlindDataClass dataClass,
                  GridType dataType,
                  size_t i, size_t valueCount, size_t valueSize)
        : metaData(reinterpret_cast<GridBlindMetaData*>(new char[sizeof(GridBlindMetaData)]))
        , order(i)// sorted id of meta data
        , size(AlignUp<NANOVDB_DATA_ALIGNMENT>(valueCount * valueSize))
    {
        std::memset(metaData, 0, sizeof(GridBlindMetaData));// zero out all meta data
        if (name.length()>=GridData::MaxNameSize) throw std::runtime_error("blind data name exceeds character limit");
        std::memcpy(metaData->mName, name.c_str(), name.length() + 1);
        metaData->mValueCount = valueCount;
        metaData->mSemantic = dataSemantic;
        metaData->mDataClass = dataClass;
        metaData->mDataType = dataType;
        metaData->mValueSize = valueSize;
        NANOVDB_ASSERT(metaData->isValid());
    }
    ~BlindMetaData(){ delete [] reinterpret_cast<char*>(metaData); }
    bool operator<(const BlindMetaData& other) const { return order < other.order; } // required by std::set
    static GridType mapToType(const std::string& name)
    {
        GridType type = GridType::Unknown;
        if ("uint32_t" == name) {
            type = GridType::UInt32;
        } else if ("float" == name) {
            type = GridType::Float;
        } else if ("vec3s"== name) {
            type = GridType::Vec3f;
        } else if ("int32" == name) {
            type = GridType::Int32;
        } else if ("int64" == name) {
            type = GridType::Int64;
        }
        return type;
    }
    static GridBlindDataSemantic mapToSemantics(const std::string& name)
    {
        GridBlindDataSemantic semantic = GridBlindDataSemantic::Unknown;
        if ("P" == name) {
            semantic = GridBlindDataSemantic::PointPosition;
        } else if ("V" == name) {
            semantic = GridBlindDataSemantic::PointVelocity;
        } else if ("Cd" == name) {
            semantic = GridBlindDataSemantic::PointColor;
        } else if ("N" == name) {
            semantic = GridBlindDataSemantic::PointNormal;
        } else if ("id" == name) {
            semantic = GridBlindDataSemantic::PointId;
        }
        return semantic;
    }
    GridBlindMetaData *metaData;
    const size_t       order, size;
}; // CreateNanoGrid::BlindMetaData

//================================================================================================

template <typename SrcGridT>
template<typename DstBuildT, typename BufferT>
typename disable_if<is_same<DstBuildT, FpN>::value ||
                    BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
CreateNanoGrid<SrcGridT>::getHandle(const BufferT& pool)
{
    this->template preProcess<DstBuildT>();
    auto handle = this->template initHandle<DstBuildT>(pool);
    this->template postProcess<DstBuildT>();
    return handle;
} // CreateNanoGrid::getHandle<T>

//================================================================================================

template <typename SrcGridT>
template<typename DstBuildT, typename OracleT, typename BufferT>
typename enable_if<is_same<DstBuildT, FpN>::value, GridHandle<BufferT>>::type
CreateNanoGrid<SrcGridT>::getHandle(const OracleT& oracle, const BufferT& pool)
{
    this->template preProcess<DstBuildT, OracleT>(oracle);
    auto handle = this->template initHandle<DstBuildT>(pool);
    this->template postProcess<DstBuildT>();
    return handle;
} // CreateNanoGrid::getHandle<FpN>

//================================================================================================

template <typename SrcGridT>
template<typename DstBuildT, typename BufferT>
typename enable_if<BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
CreateNanoGrid<SrcGridT>::getHandle(uint32_t channels,
                                    bool includeStats,
                                    bool includeTiles,
                                    const BufferT &pool)
{
    mIncludeStats = includeStats;
    mIncludeTiles = includeTiles;
    this->template preProcess<DstBuildT>(channels);
    auto handle = this->template initHandle<DstBuildT>(pool);
    this->template postProcess<DstBuildT>(channels);
    return handle;
}// CreateNanoGrid::getHandle<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT, typename BufferT>
GridHandle<BufferT> CreateNanoGrid<SrcGridT>::initHandle(const BufferT& pool)
{
    mOffset.grid  = 0;// grid is always stored at the start of the buffer!
    mOffset.tree  = NanoGrid<DstBuildT>::memUsage(); // grid ends and tree begins
    mOffset.root  = mOffset.tree  + NanoTree<DstBuildT>::memUsage(); // tree ends and root node begins
    mOffset.upper = mOffset.root  + NanoRoot<DstBuildT>::memUsage(mSrcNodeAcc.root().getTableSize()); // root node ends and upper internal nodes begin
    mOffset.lower = mOffset.upper + NanoUpper<DstBuildT>::memUsage()*mSrcNodeAcc.nodeCount(2); // upper internal nodes ends and lower internal nodes begin
    mOffset.leaf  = mOffset.lower + NanoLower<DstBuildT>::memUsage()*mSrcNodeAcc.nodeCount(1); // lower internal nodes ends and leaf nodes begin
    mOffset.meta  = mOffset.leaf  + mLeafNodeSize;// leaf nodes end and blind meta data begins
    mOffset.blind = mOffset.meta  + sizeof(GridBlindMetaData)*mBlindMetaData.size(); // meta data ends and blind data begins
    mOffset.size  = mOffset.blind;// end of buffer
    for (const auto& b : mBlindMetaData) mOffset.size += b.size; // accumulate all the blind data

    auto buffer = BufferT::create(mOffset.size, &pool);
    mBufferPtr = buffer.data();

    // Concurrent processing of all tree levels!
    invoke( [&](){this->template processLeafs<DstBuildT>();},
            [&](){this->template processInternalNodes<DstBuildT, 1>();},
            [&](){this->template processInternalNodes<DstBuildT, 2>();},
            [&](){this->template processRoot<DstBuildT>();},
            [&](){this->template processTree<DstBuildT>();},
            [&](){this->template processGrid<DstBuildT>();} );

    return GridHandle<BufferT>(std::move(buffer));
} // CreateNanoGrid::initHandle

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename disable_if<is_same<FpN, DstBuildT>::value || BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::preProcess()
{
    if (const uint64_t pointCount = this->countPoints()) {
#if defined(NANOVDB_USE_OPENVDB)
        if constexpr(is_same<openvdb::tools::PointIndexGrid, SrcGridT>::value) {
            if (!mBlindMetaData.empty()) throw std::runtime_error("expected no blind meta data");
            this->addBlindData("index",
                               GridBlindDataSemantic::PointId,
                               GridBlindDataClass::IndexArray,
                               GridType::UInt32,
                               pointCount,
                               sizeof(uint32_t));
        } else if constexpr(is_same<openvdb::points::PointDataGrid, SrcGridT>::value) {
            if (!mBlindMetaData.empty()) throw std::runtime_error("expected no blind meta data");
            auto &srcLeaf = mSrcNodeAcc.template node<0>(0);
            const auto& attributeSet = srcLeaf.attributeSet();
            const auto& descriptor = attributeSet.descriptor();
            const auto& nameMap = descriptor.map();
            for (auto it = nameMap.begin(); it != nameMap.end(); ++it) {
                const size_t index = it->second;
                auto& attArray = srcLeaf.constAttributeArray(index);
                mBlindMetaData.emplace(it->first, // name used to derive semantics
                                       descriptor.valueType(index), // type
                                       it->first == "id" ? GridBlindDataClass::IndexArray : GridBlindDataClass::AttributeArray, // class
                                       index, // order
                                       pointCount, // element count
                                       attArray.valueTypeSize()); // element size
            }
        }
#endif// end NANOVDB_USE_OPENVDB
    }
    if (mSrcNodeAcc.hasLongGridName()) {
        this->addBlindData("grid name",
                           GridBlindDataSemantic::Unknown,
                           GridBlindDataClass::GridName,
                           GridType::Unknown,
                           mSrcNodeAcc.getName().length() + 1, 1);
    }
    mLeafNodeSize = mSrcNodeAcc.nodeCount(0)*NanoLeaf<DstBuildT>::DataType::memUsage();
}// CreateNanoGrid::preProcess<T>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT, typename OracleT>
inline typename enable_if<is_same<FpN, DstBuildT>::value>::type
CreateNanoGrid<SrcGridT>::preProcess(OracleT oracle)
{
    static_assert(is_same<float, SrcValueT>::value, "preProcess<FpN>: expected SrcValueT == float");

    const size_t leafCount = mSrcNodeAcc.nodeCount(0);
    if (leafCount==0) {
        mLeafNodeSize = 0u;
        return;
    }
    mCodec.reset(new Codec[leafCount]);

    if constexpr(is_same<AbsDiff, OracleT>::value) {
        if (!oracle) oracle.init(mSrcNodeAcc.gridClass(), mSrcNodeAcc.root().background());
    }

    DitherLUT lut(mDitherOn);
    forEach(0, leafCount, 4, [&](const Range1D &r) {
        for (auto i=r.begin(); i!=r.end(); ++i) {
            const auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            float &min = mCodec[i].min = std::numeric_limits<float>::max();
            float &max = mCodec[i].max = -min;
            for (int j=0; j<512; ++j) {
                float v = srcLeaf.getValue(j);
                if (v<min) min = v;
                if (v>max) max = v;
            }
            const float range = max - min;
            uint8_t &logBitWidth = mCodec[i].log2 = 0;// 0,1,2,3,4 => 1,2,4,8,16 bits
            while (range > 0.0f && logBitWidth < 4u) {
                const uint32_t mask = (uint32_t(1) << (uint32_t(1) << logBitWidth)) - 1u;
                const float encode  = mask/range;
                const float decode  = range/mask;
                int j = 0;
                do {
                    const float exact = srcLeaf.getValue(j);//data[j];// exact value
                    const uint32_t code = uint32_t(encode*(exact - min) + lut(j));
                    const float approx = code * decode + min;// approximate value
                    j += oracle(exact, approx) ? 1 : 513;
                } while(j < 512);
                if (j == 512) break;
                ++logBitWidth;
            }
        }
    });

    auto getOffset = [&](size_t i){
        --i;
        return mCodec[i].offset +  NanoLeaf<DstBuildT>::DataType::memUsage(1u << mCodec[i].log2);
    };
    mCodec[0].offset = NanoGrid<FpN>::memUsage() +
                       NanoTree<FpN>::memUsage() +
                       NanoRoot<FpN>::memUsage(mSrcNodeAcc.root().getTableSize()) +
                       NanoUpper<FpN>::memUsage()*mSrcNodeAcc.nodeCount(2) +
                       NanoLower<FpN>::memUsage()*mSrcNodeAcc.nodeCount(1);
    for (size_t i=1; i<leafCount; ++i) mCodec[i].offset = getOffset(i);
    mLeafNodeSize = getOffset(leafCount);

    if (mVerbose) {
        uint32_t counters[5+1] = {0};
        ++counters[mCodec[0].log2];
        for (size_t i=1; i<leafCount; ++i) ++counters[mCodec[i].log2];
        std::cout << "\n" << oracle << std::endl;
        std::cout << "Dithering: " << (mDitherOn ? "enabled" : "disabled") << std::endl;
        float avg = 0.0f;
        for (uint32_t i=0; i<=5; ++i) {
            if (uint32_t n = counters[i]) {
                avg += n * float(1 << i);
                printf("%2i bits: %6u leaf nodes, i.e. %4.1f%%\n",1<<i, n, 100.0f*n/float(leafCount));
            }
        }
        printf("%4.1f bits per value on average\n", avg/float(leafCount));
    }

    if (mSrcNodeAcc.hasLongGridName()) {
        this->addBlindData("grid name",
                           GridBlindDataSemantic::Unknown,
                           GridBlindDataClass::GridName,
                           GridType::Unknown,
                           mSrcNodeAcc.getName().length() + 1, 1);
    }
}// CreateNanoGrid::preProcess<FpN>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT, int LEVEL>
inline typename enable_if<BuildTraits<DstBuildT>::is_index, uint64_t>::type
CreateNanoGrid<SrcGridT>::countTileValues(uint64_t valueCount)
{
    const uint64_t stats = mIncludeStats ? 4u : 0u;// minimum, maximum, average, and deviation
    mValIdx[LEVEL].clear();
    mValIdx[LEVEL].resize(mSrcNodeAcc.nodeCount(LEVEL) + 1, stats);// minimum 1 entry
    forEach(1, mValIdx[LEVEL].size(), 8, [&](const Range1D& r){
        for (auto i = r.begin(); i!=r.end(); ++i) {
            auto &srcNode = mSrcNodeAcc.template node<LEVEL>(i-1);
            if constexpr(BuildTraits<DstBuildT>::is_onindex) {// resolved at compile time
                mValIdx[LEVEL][i] += srcNode.getValueMask().countOn();
            } else {
                static const uint64_t maxTileCount = uint64_t(1u) << 3*srcNode.LOG2DIM;
                mValIdx[LEVEL][i] += maxTileCount - srcNode.getChildMask().countOn();
            }
        }
    });
    mValIdx[LEVEL][0] = valueCount;
    for (size_t i=1; i<mValIdx[LEVEL].size(); ++i) mValIdx[LEVEL][i] += mValIdx[LEVEL][i-1];// pre-fixed sum
    return mValIdx[LEVEL].back();
}// CreateNanoGrid::countTileValues<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_index, uint64_t>::type
CreateNanoGrid<SrcGridT>::countValues()
{
    const uint64_t stats = mIncludeStats ? 4u : 0u;// minimum, maximum, average, and deviation
    uint64_t valueCount = 1u;// offset 0 corresponds to the background value
    if (mIncludeTiles) {
        if constexpr(BuildTraits<DstBuildT>::is_onindex) {
            for (auto it = mSrcNodeAcc.root().cbeginValueOn(); it; ++it) ++valueCount;
        } else {
            for (auto it = mSrcNodeAcc.root().cbeginValueAll(); it; ++it) ++valueCount;
        }
        valueCount += stats;// optionally append stats for the root node
        valueCount = countTileValues<DstBuildT, 2>(valueCount);
        valueCount = countTileValues<DstBuildT, 1>(valueCount);
    }
    mValIdx[0].clear();
    mValIdx[0].resize(mSrcNodeAcc.nodeCount(0) + 1, 512u + stats);// minimum 1 entry
    if constexpr(BuildTraits<DstBuildT>::is_onindex) {
        forEach(1, mValIdx[0].size(), 8, [&](const Range1D& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                mValIdx[0][i] = stats;
                mValIdx[0][i] += mSrcNodeAcc.template node<0>(i-1).getValueMask().countOn();
            }
        });
    }
    mValIdx[0][0] = valueCount;
    prefixSum(mValIdx[0], true);// inclusive prefix sum
    return mValIdx[0].back();
}// CreateNanoGrid::countValues<ValueIndex or ValueOnIndex>()

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::preProcess(uint32_t channels)
{
    const uint64_t valueCount = this->template countValues<DstBuildT>();
    mLeafNodeSize = mSrcNodeAcc.nodeCount(0)*NanoLeaf<DstBuildT>::DataType::memUsage();

    uint32_t order = mBlindMetaData.size();
    for (uint32_t i=0; i<channels; ++i) {
        mBlindMetaData.emplace("channel_"+std::to_string(i),
                               toStr(mapToGridType<SrcValueT>()),
                               GridBlindDataClass::AttributeArray,
                               order++,
                               valueCount,
                               sizeof(SrcValueT));
    }
    if (mSrcNodeAcc.hasLongGridName()) {
        this->addBlindData("grid name",
                           GridBlindDataSemantic::Unknown,
                           GridBlindDataClass::GridName,
                           GridType::Unknown,
                           mSrcNodeAcc.getName().length() + 1, 1);
    }
}// preProcess<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename disable_if<BuildTraits<DstBuildT>::is_special>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    using DstDataT  = typename NanoLeaf<DstBuildT>::DataType;
    using DstValueT = typename DstDataT::ValueType;
    static_assert(DstDataT::FIXED_SIZE, "Expected destination LeafNode<T> to have fixed size");
    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            if (DstDataT::padding()>0u) {
                // Cast to void* to avoid compiler warning about missing trivial copy-assignment
                std::memset(reinterpret_cast<void*>(dstData), 0, DstDataT::memUsage());
            } else {
                dstData->mBBoxDif[0] = dstData->mBBoxDif[1] = dstData->mBBoxDif[2] = 0u;
                dstData->mFlags = 0u;// enable rendering, no bbox, no stats
                dstData->mMinimum = dstData->mMaximum = typename DstDataT::ValueType();
                dstData->mAverage = dstData->mStdDevi = 0;
            }
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
            DstValueT *dst = dstData->mValues;
            if constexpr(is_same<DstValueT, SrcValueT>::value && SrcNodeAccT::IS_OPENVDB) {
                const SrcValueT *src = srcLeaf.buffer().data();
                for (auto *end = dst + 512u; dst != end; dst += 4, src += 4) {
                    dst[0] = src[0]; // copy *all* voxel values in sets of four, i.e. loop-unrolling
                    dst[1] = src[1];
                    dst[2] = src[2];
                    dst[3] = src[3];
                }
            } else {
                for (uint32_t j=0; j<512u; ++j) *dst++ = static_cast<DstValueT>(srcLeaf.getValue(j));
            }
        }
    });
} // CreateNanoGrid::processLeafs<T>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    using DstDataT  = typename NanoLeaf<DstBuildT>::DataType;
    static_assert(DstDataT::FIXED_SIZE, "Expected destination LeafNode<ValueIndex> to have fixed size");
    static_assert(DstDataT::padding()==0u, "Expected leaf nodes to have no padding");

    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        const uint8_t flags  = mIncludeStats ? 16u : 0u;// 4th bit indicates stats
        DstDataT *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();// fixed size
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mBBoxDif[0] = dstData->mBBoxDif[1] = dstData->mBBoxDif[2] = 0u;
            dstData->mFlags = flags;
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
            dstData->mOffset = mValIdx[0][i];
            if constexpr(BuildTraits<DstBuildT>::is_onindex) {
                const uint64_t *w = dstData->mValueMask.words();
#ifdef USE_OLD_VALUE_ON_INDEX
                int32_t sum = CountOn(*w++);
                uint8_t *p = reinterpret_cast<uint8_t*>(&dstData->mPrefixSum), *q = p + 7;
                for (int j=0; j<7; ++j) {
                    *p++ = sum & 255u;
                    *q |= (sum >> 8) << j;
                    sum += CountOn(*w++);
                }
#else
                uint64_t &prefixSum = dstData->mPrefixSum, sum = CountOn(*w++);
                prefixSum = sum;
                for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
                    sum += CountOn(*w++);
                    prefixSum |= sum << n;// each pre-fixed sum is encoded in 9 bits
                }
#endif
            } else {
                dstData->mPrefixSum = 0u;
            }
            if constexpr(BuildTraits<DstBuildT>::is_indexmask) dstData->mMask = dstData->mValueMask;
        }
    });
} // CreateNanoGrid::processLeafs<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<is_same<ValueMask, DstBuildT>::value>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    using DstDataT = typename NanoLeaf<ValueMask>::DataType;
    static_assert(DstDataT::FIXED_SIZE, "Expected destination LeafNode<ValueMask> to have fixed size");
    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            if (DstDataT::padding()>0u) {
                // Cast to void* to avoid compiler warning about missing trivial copy-assignment
                std::memset(reinterpret_cast<void*>(dstData), 0, DstDataT::memUsage());
            } else {
                dstData->mBBoxDif[0] = dstData->mBBoxDif[1] = dstData->mBBoxDif[2] = 0u;
                dstData->mFlags = 0u;// enable rendering, no bbox, no stats
                dstData->mPadding[0] = dstData->mPadding[1] = 0u;
            }
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
        }
    });
} // CreateNanoGrid::processLeafs<ValueMask>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<is_same<bool, DstBuildT>::value>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    using DstDataT = typename NanoLeaf<bool>::DataType;
    static_assert(DstDataT::FIXED_SIZE, "Expected destination LeafNode<bool> to have fixed size");
    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            if (DstDataT::padding()>0u) {
                // Cast to void* to avoid compiler warning about missing trivial copy-assignment
                std::memset(reinterpret_cast<void*>(dstData), 0, DstDataT::memUsage());
            } else {
                dstData->mBBoxDif[0] = dstData->mBBoxDif[1] = dstData->mBBoxDif[2] = 0u;
                dstData->mFlags = 0u;// enable rendering, no bbox, no stats
            }
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
            if constexpr(!is_same<bool, SrcBuildT>::value) {
                for (int j=0; j<512; ++j) dstData->mValues.set(j, static_cast<bool>(srcLeaf.getValue(j)));
            } else if constexpr(SrcNodeAccT::IS_OPENVDB) {
                dstData->mValues = *reinterpret_cast<const Mask<3>*>(srcLeaf.buffer().data());
            } else if constexpr(SrcNodeAccT::IS_NANOVDB) {
                dstData->mValues = srcLeaf.data()->mValues;
            } else {// build::Leaf
                dstData->mValues = srcLeaf.mValues; // copy value mask
            }
        }
    });
} // CreateNanoGrid::processLeafs<bool>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_FpX>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    using DstDataT = typename NanoLeaf<DstBuildT>::DataType;
    static_assert(DstDataT::FIXED_SIZE, "Expected destination LeafNode<Fp4|Fp8|Fp16> to have fixed size");
    using ArrayT = typename DstDataT::ArrayType;
    static_assert(is_same<float, SrcValueT>::value, "Expected ValueT == float");
    using FloatT = typename std::conditional<DstDataT::bitWidth()>=16, double, float>::type;// 16 compression and higher requires double
    static constexpr FloatT UNITS = FloatT((1 << DstDataT::bitWidth()) - 1);// # of unique non-zero values
    DitherLUT lut(mDitherOn);

    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            if (DstDataT::padding()>0u) {
                // Cast to void* to avoid compiler warning about missing trivial copy-assignment
                std::memset(reinterpret_cast<void*>(dstData), 0, DstDataT::memUsage());
            } else {
                dstData->mFlags = dstData->mBBoxDif[2] = dstData->mBBoxDif[1] = dstData->mBBoxDif[0] = 0u;
                dstData->mDev = dstData->mAvg = dstData->mMax = dstData->mMin = 0u;
            }
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
            // compute extrema values
            float min = std::numeric_limits<float>::max(), max = -min;
            for (uint32_t j=0; j<512u; ++j) {
                const float v = srcLeaf.getValue(j);
                if (v < min) min = v;
                if (v > max) max = v;
            }
            dstData->init(min, max, DstDataT::bitWidth());
            // perform quantization relative to the values in the current leaf node
            const FloatT encode = UNITS/(max-min);
            uint32_t offset = 0;
            auto quantize = [&]()->ArrayT{
                const ArrayT tmp = static_cast<ArrayT>(encode * (srcLeaf.getValue(offset) - min) + lut(offset));
                ++offset;
                return tmp;
            };
            auto *code = reinterpret_cast<ArrayT*>(dstData->mCode);
            if (is_same<Fp4, DstBuildT>::value) {// resolved at compile-time
                for (uint32_t j=0; j<128u; ++j) {
                    auto tmp = quantize();
                    *code++  = quantize() << 4 | tmp;
                    tmp      = quantize();
                    *code++  = quantize() << 4 | tmp;
                }
            } else {
                for (uint32_t j=0; j<128u; ++j) {
                    *code++ = quantize();
                    *code++ = quantize();
                    *code++ = quantize();
                    *code++ = quantize();
                }
            }
        }
    });
} // CreateNanoGrid::processLeafs<Fp4, Fp8, Fp16>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<is_same<FpN, DstBuildT>::value>::type
CreateNanoGrid<SrcGridT>::processLeafs()
{
    static_assert(is_same<float, SrcValueT>::value, "Expected SrcValueT == float");
    DitherLUT lut(mDitherOn);
    forEach(0, mSrcNodeAcc.nodeCount(0), 8, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto &srcLeaf = mSrcNodeAcc.template node<0>(i);
            auto *dstData = this->template dstNode<DstBuildT,0>(i)->data();
            dstData->mBBoxMin = srcLeaf.origin(); // copy origin of node
            dstData->mBBoxDif[0] = dstData->mBBoxDif[1] = dstData->mBBoxDif[2] = 0u;
            const uint8_t logBitWidth = mCodec[i].log2;
            dstData->mFlags = logBitWidth << 5;// pack logBitWidth into 3 MSB of mFlag
            dstData->mValueMask = srcLeaf.getValueMask(); // copy value mask
            const float min = mCodec[i].min, max = mCodec[i].max;
            dstData->init(min, max, uint8_t(1) << logBitWidth);
            // perform quantization relative to the values in the current leaf node
            uint32_t offset = 0;
            float encode = 0.0f;
            auto quantize = [&]()->uint8_t{
                const uint8_t tmp = static_cast<uint8_t>(encode * (srcLeaf.getValue(offset) - min) + lut(offset));
                ++offset;
                return tmp;
            };
            auto *dst = reinterpret_cast<uint8_t*>(dstData+1);
            switch (logBitWidth) {
                case 0u: {// 1 bit
                    encode = 1.0f/(max - min);
                    for (int j=0; j<64; ++j) {
                        uint8_t a = 0;
                        for (int k=0; k<8; ++k) a |= quantize() << k;
                        *dst++ = a;
                    }
                }
                break;
                case 1u: {// 2 bits
                    encode = 3.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = quantize();
                        a     |= quantize() << 2;
                        a     |= quantize() << 4;
                        *dst++ = quantize() << 6 | a;
                    }
                }
                break;
                case 2u: {// 4 bits
                    encode = 15.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = quantize();
                        *dst++ = quantize() << 4 | a;
                        a      = quantize();
                        *dst++ = quantize() << 4 | a;
                    }
                }
                break;
                case 3u: {// 8 bits
                    encode = 255.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        *dst++ = quantize();
                        *dst++ = quantize();
                        *dst++ = quantize();
                        *dst++ = quantize();
                    }
                }
                break;
                default: {// 16 bits - special implementation using higher bit-precision
                    auto *dst = reinterpret_cast<uint16_t*>(dstData+1);
                    const double encode = 65535.0/(max - min);// note that double is required!
                    for (int j=0; j<128; ++j) {
                        *dst++ = uint16_t(encode * (srcLeaf.getValue(offset) - min) + lut(offset)); ++offset;
                        *dst++ = uint16_t(encode * (srcLeaf.getValue(offset) - min) + lut(offset)); ++offset;
                        *dst++ = uint16_t(encode * (srcLeaf.getValue(offset) - min) + lut(offset)); ++offset;
                        *dst++ = uint16_t(encode * (srcLeaf.getValue(offset) - min) + lut(offset)); ++offset;
                    }
                }
            }// end switch
        }
    });// kernel
} // CreateNanoGrid::processLeafs<FpN>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT, int LEVEL>
inline typename enable_if<!BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::processInternalNodes()
{
    using DstNodeT  = typename NanoNode<DstBuildT, LEVEL>::type;
    using DstValueT = typename DstNodeT::ValueType;
    using DstChildT = typename NanoNode<DstBuildT, LEVEL-1>::type;
    static_assert(LEVEL == 1 || LEVEL == 2, "Expected internal node");

    const uint64_t nodeCount = mSrcNodeAcc.nodeCount(LEVEL);
    if (nodeCount > 0) {// compute and temporarily encode IDs of child nodes
        uint64_t childCount = 0;
        auto *dstData = this->template dstNode<DstBuildT,LEVEL>(0)->data();
        for (uint64_t i=0; i<nodeCount; ++i) {
            dstData[i].mFlags = childCount;
            childCount += mSrcNodeAcc.template node<LEVEL>(i).getChildMask().countOn();
        }
    }

    forEach(0, nodeCount, 4, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,LEVEL>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcNode  = mSrcNodeAcc.template node<LEVEL>(i);
            uint64_t childID = dstData->mFlags;
            if (DstNodeT::DataType::padding()>0u) {
                // Cast to void* to avoid compiler warning about missing trivial copy-assignment
                std::memset(reinterpret_cast<void*>(dstData), 0, DstNodeT::memUsage());
            } else {
                dstData->mFlags = 0;// enable rendering, no bbox, no stats
                dstData->mMinimum = dstData->mMaximum = typename DstNodeT::ValueType();
                dstData->mAverage = dstData->mStdDevi = 0;
            }
            dstData->mBBox[0]   = srcNode.origin(); // copy origin of node
            dstData->mValueMask = srcNode.getValueMask(); // copy value mask
            dstData->mChildMask = srcNode.getChildMask(); // copy child mask
            for (auto it = srcNode.cbeginChildAll(); it; ++it) {
                SrcValueT value{}; // default initialization
                if (it.probeChild(value)) {
                    DstChildT *dstChild = this->template dstNode<DstBuildT,LEVEL-1>(childID++);// might be Leaf<FpN>
                    dstData->setChild(it.pos(), dstChild);
                } else {
                    dstData->setValue(it.pos(), static_cast<DstValueT>(value));
                }
            }
        }
    });
} // CreateNanoGrid::processInternalNodes<T>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT, int LEVEL>
inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::processInternalNodes()
{
    using DstNodeT  = typename NanoNode<DstBuildT, LEVEL>::type;
    using DstChildT = typename NanoNode<DstBuildT, LEVEL-1>::type;
    static_assert(LEVEL == 1 || LEVEL == 2, "Expected internal node");
    static_assert(DstNodeT::DataType::padding()==0u, "Expected internal nodes to have no padding");

    const uint64_t nodeCount = mSrcNodeAcc.nodeCount(LEVEL);
    if (nodeCount > 0) {// compute and temporarily encode IDs of child nodes
        uint64_t childCount = 0;
        auto *dstData = this->template dstNode<DstBuildT,LEVEL>(0)->data();
        for (uint64_t i=0; i<nodeCount; ++i) {
            dstData[i].mFlags = childCount;
            childCount += mSrcNodeAcc.template node<LEVEL>(i).getChildMask().countOn();
        }
    }

    forEach(0, nodeCount, 4, [&](const Range1D& r) {
        auto *dstData = this->template dstNode<DstBuildT,LEVEL>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto &srcNode  = mSrcNodeAcc.template node<LEVEL>(i);
            uint64_t childID = dstData->mFlags;
            dstData->mFlags = 0u;
            dstData->mBBox[0]   = srcNode.origin(); // copy origin of node
            dstData->mValueMask = srcNode.getValueMask(); // copy value mask
            dstData->mChildMask = srcNode.getChildMask(); // copy child mask
            uint64_t n = mIncludeTiles ? mValIdx[LEVEL][i] : 0u;
            for (auto it = srcNode.cbeginChildAll(); it; ++it) {
                SrcValueT value;
                if (it.probeChild(value)) {
                    DstChildT *dstChild = this->template dstNode<DstBuildT,LEVEL-1>(childID++);// might be Leaf<FpN>
                    dstData->setChild(it.pos(), dstChild);
                } else {
                    uint64_t m = 0u;
                    if (mIncludeTiles && !((BuildTraits<DstBuildT>::is_onindex) && dstData->mValueMask.isOff(it.pos()))) m = n++;
                    dstData->setValue(it.pos(), m);
                }
            }
            if (mIncludeTiles && mIncludeStats) {// stats are always placed after the tile values
                dstData->mMinimum = n++;
                dstData->mMaximum = n++;
                dstData->mAverage = n++;
                dstData->mStdDevi = n++;
            } else {// if not tiles or stats set stats to the background offset
                dstData->mMinimum = 0u;
                dstData->mMaximum = 0u;
                dstData->mAverage = 0u;
                dstData->mStdDevi = 0u;
            }
        }
    });
} // CreateNanoGrid::processInternalNodes<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<!BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::processRoot()
{
    using DstRootT  = NanoRoot<DstBuildT>;
    using DstValueT = typename DstRootT::ValueType;
    auto &srcRoot = mSrcNodeAcc.root();
    auto *dstData = this->template dstRoot<DstBuildT>()->data();
    const uint32_t tableSize = srcRoot.getTableSize();
    // Cast to void* to avoid compiler warning about missing trivial copy-assignment
    if (DstRootT::DataType::padding()>0) std::memset(reinterpret_cast<void*>(dstData), 0, DstRootT::memUsage(tableSize));
    dstData->mTableSize = tableSize;
    dstData->mMinimum = dstData->mMaximum = dstData->mBackground = srcRoot.background();
    dstData->mBBox = CoordBBox(); // // set to an empty bounding box
    if (tableSize==0) return;
    auto *dstChild = this->template dstNode<DstBuildT, 2>(0);// fixed size and linear in memory
    auto *dstTile  = dstData->tile(0);// fixed size and linear in memory
    for (auto it = srcRoot.cbeginChildAll(); it; ++it, ++dstTile) {
        SrcValueT value;
        if (it.probeChild(value)) {
            dstTile->setChild(it.getCoord(), dstChild++, dstData);
        } else {
            dstTile->setValue(it.getCoord(), it.isValueOn(), static_cast<DstValueT>(value));
        }
    }
} // CreateNanoGrid::processRoot<T>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::processRoot()
{
    using DstRootT  = NanoRoot<DstBuildT>;
    auto &srcRoot = mSrcNodeAcc.root();
    auto *dstData = this->template dstRoot<DstBuildT>()->data();
    const uint32_t tableSize = srcRoot.getTableSize();
    // Cast to void* to avoid compiler warning about missing trivial copy-assignment
    if (DstRootT::DataType::padding()>0) std::memset(reinterpret_cast<void*>(dstData), 0, DstRootT::memUsage(tableSize));
    dstData->mTableSize = tableSize;
    dstData->mBackground = 0u;
    uint64_t valueCount = 0u;// the first entry is always the background value
    dstData->mBBox = CoordBBox(); // set to an empty/invalid bounding box

    if (tableSize>0) {
        auto *dstChild = this->template dstNode<DstBuildT, 2>(0);// fixed size and linear in memory
        auto *dstTile  = dstData->tile(0);// fixed size and linear in memory
        for (auto it = srcRoot.cbeginChildAll(); it; ++it, ++dstTile) {
            SrcValueT tmp;
            if (it.probeChild(tmp)) {
                dstTile->setChild(it.getCoord(), dstChild++, dstData);
            } else {
                dstTile->setValue(it.getCoord(), it.isValueOn(), 0u);
                if (mIncludeTiles && !((BuildTraits<DstBuildT>::is_onindex) && !dstTile->state)) dstTile->value = ++valueCount;
            }
        }
    }
    if (mIncludeTiles && mIncludeStats) {// stats are always placed after the tile values
        dstData->mMinimum = ++valueCount;
        dstData->mMaximum = ++valueCount;
        dstData->mAverage = ++valueCount;
        dstData->mStdDevi = ++valueCount;
    } else if (dstData->padding()==0) {
        dstData->mMinimum = 0u;
        dstData->mMaximum = 0u;
        dstData->mAverage = 0u;
        dstData->mStdDevi = 0u;
    }
} // CreateNanoGrid::processRoot<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
void CreateNanoGrid<SrcGridT>::processTree()
{
    const uint64_t nodeCount[3] = {mSrcNodeAcc.nodeCount(0), mSrcNodeAcc.nodeCount(1), mSrcNodeAcc.nodeCount(2)};
    auto *dstTree = this->template dstTree<DstBuildT>();
    auto *dstData = dstTree->data();
    dstData->setRoot( this->template dstRoot<DstBuildT>() );

    dstData->setFirstNode(nodeCount[2] ? this->template dstNode<DstBuildT, 2>(0) : nullptr);
    dstData->setFirstNode(nodeCount[1] ? this->template dstNode<DstBuildT, 1>(0) : nullptr);
    dstData->setFirstNode(nodeCount[0] ? this->template dstNode<DstBuildT, 0>(0) : nullptr);

    dstData->mNodeCount[0] = static_cast<uint32_t>(nodeCount[0]);
    dstData->mNodeCount[1] = static_cast<uint32_t>(nodeCount[1]);
    dstData->mNodeCount[2] = static_cast<uint32_t>(nodeCount[2]);

    // Count number of active leaf level tiles
    dstData->mTileCount[0] = reduce(Range1D(0,nodeCount[1]), uint32_t(0), [&](Range1D &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mSrcNodeAcc.template node<1>(i).getValueMask().countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active lower internal node tiles
    dstData->mTileCount[1] = reduce(Range1D(0,nodeCount[2]), uint32_t(0), [&](Range1D &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mSrcNodeAcc.template node<2>(i).getValueMask().countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active upper internal node tiles
    dstData->mTileCount[2] = 0;
    for (auto it = mSrcNodeAcc.root().cbeginValueOn(); it; ++it) dstData->mTileCount[2] += 1;

    // Count number of active voxels
    dstData->mVoxelCount = reduce(Range1D(0, nodeCount[0]), uint64_t(0), [&](Range1D &r, uint64_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mSrcNodeAcc.template node<0>(i).getValueMask().countOn();
        return sum;}, std::plus<uint64_t>());

    dstData->mVoxelCount += uint64_t(dstData->mTileCount[0]) <<  9;// = 3 * 3
    dstData->mVoxelCount += uint64_t(dstData->mTileCount[1]) << 21;// = 3 * (3+4)
    dstData->mVoxelCount += uint64_t(dstData->mTileCount[2]) << 36;// = 3 * (3+4+5)

} // CreateNanoGrid::processTree

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
void CreateNanoGrid<SrcGridT>::processGrid()
{
    auto* dstData = this->template dstGrid<DstBuildT>()->data();
    dstData->init({GridFlags::IsBreadthFirst}, mOffset.size, mSrcNodeAcc.map(),
                  mapToGridType<DstBuildT>(), mapToGridClass<DstBuildT>(mSrcNodeAcc.gridClass()));
     dstData->mBlindMetadataCount = static_cast<uint32_t>(mBlindMetaData.size());
     dstData->mData1 = this->valueCount();

    if (!isValid(dstData->mGridType, dstData->mGridClass)) {
        std::stringstream ss;
        ss << "Invalid combination of GridType("<<int(dstData->mGridType)
           << ") and GridClass("<<int(dstData->mGridClass)<<"). See NanoVDB.h for details!";
        throw std::runtime_error(ss.str());
    }

    std::memset(dstData->mGridName, '\0', GridData::MaxNameSize);//overwrite mGridName
    strncpy(dstData->mGridName, mSrcNodeAcc.getName().c_str(), GridData::MaxNameSize-1);
    if (mSrcNodeAcc.hasLongGridName()) dstData->setLongGridNameOn();// grid name is long so store it as blind data

    // Partially process blind meta data - they will be complete in postProcess
    if (mBlindMetaData.size()>0) {
        auto *metaData = this->dstMeta(0);
        dstData->mBlindMetadataOffset = PtrDiff(metaData, dstData);
        dstData->mBlindMetadataCount = static_cast<uint32_t>(mBlindMetaData.size());
        char *blindData = PtrAdd<char>(mBufferPtr, mOffset.blind);
        for (const auto &b : mBlindMetaData) {
            std::memcpy(metaData, b.metaData, sizeof(GridBlindMetaData));
            metaData->setBlindData(blindData);// sets metaData.mOffset
            if (metaData->mDataClass == GridBlindDataClass::GridName) strcpy(blindData, mSrcNodeAcc.getName().c_str());
            ++metaData;
            blindData += b.size;
        }
        mBlindMetaData.clear();
    }
} // CreateNanoGrid::processGrid

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename disable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::postProcess()
{
    if constexpr(is_same<FpN, DstBuildT>::value) mCodec.reset();
    auto *dstGrid = this->template dstGrid<DstBuildT>();
    gridStats(*dstGrid, mStats);
#if defined(NANOVDB_USE_OPENVDB)
    auto *metaData = this->dstMeta(0);
    if constexpr(is_same<openvdb::tools::PointIndexGrid, SrcGridT>::value ||
                 is_same<openvdb::points::PointDataGrid, SrcGridT>::value) {
        static_assert(is_same<DstBuildT, uint32_t>::value, "expected DstBuildT==uint32_t");
        auto *dstData0 = this->template dstNode<DstBuildT,0>(0)->data();
        dstData0->mMinimum = 0; // start of prefix sum
        dstData0->mMaximum = dstData0->mValues[511u];
        for (uint32_t i=1, n=mSrcNodeAcc.nodeCount(0); i<n; ++i) {
            auto *dstData1 = dstData0 + 1;
            dstData1->mMinimum = dstData0->mMinimum + dstData0->mMaximum;
            dstData1->mMaximum = dstData1->mValues[511u];
            dstData0 = dstData1;
        }
        for (size_t i = 0, n = dstGrid->blindDataCount(); i < n; ++i, ++metaData) {
            if constexpr(is_same<openvdb::tools::PointIndexGrid, SrcGridT>::value) {
                if (metaData->mDataClass != GridBlindDataClass::IndexArray) continue;
                if (metaData->mDataType == GridType::UInt32) {
                    uint32_t *blindData = const_cast<uint32_t*>(metaData->template getBlindData<uint32_t>());
                    forEach(0, mSrcNodeAcc.nodeCount(0), 16, [&](const auto& r) {
                        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
                        for (auto j = r.begin(); j != r.end(); ++j, ++dstData) {
                            uint32_t* p = blindData + dstData->mMinimum;
                            for (uint32_t idx : mSrcNodeAcc.template node<0>(j).indices()) *p++ = idx;
                        }
                    });
                }
            } else {// if constexpr(is_same<openvdb::points::PointDataGrid, SrcGridT>::value)
                if (metaData->mDataClass != GridBlindDataClass::AttributeArray) continue;
                if (auto *blindData = dstGrid->template getBlindData<float>(i)) {
                    this->template copyPointAttribute<DstBuildT>(i, blindData);
                } else if (auto *blindData = dstGrid->template getBlindData<nanovdb::Vec3f>(i)) {
                    this->template copyPointAttribute<DstBuildT>(i, reinterpret_cast<openvdb::Vec3f*>(blindData));
                } else if (auto *blindData = dstGrid->template getBlindData<int32_t>(i)) {
                    this->template copyPointAttribute<DstBuildT>(i, blindData);
                } else if (auto *blindData = dstGrid->template getBlindData<int64_t>(i)) {
                    this->template copyPointAttribute<DstBuildT>(i, blindData);
                } else {
                    std::cerr << "unsupported point attribute \"" << toStr(metaData->mDataType) << "\"\n";
                }
            }// if
        }// loop
    } else { // if
        (void)metaData;
    }
#endif
    updateChecksum(*(this->template dstGrid<DstBuildT>()), mChecksum);
}// CreateNanoGrid::postProcess<T>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
inline typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::postProcess(uint32_t channels)
{
    const std::string typeName = toStr(mapToGridType<SrcValueT>());
    const uint64_t valueCount = this->valueCount();
    const auto *dstGrid = this->template dstGrid<DstBuildT>();
    for (uint32_t i=0; i<channels; ++i) {
        const std::string name = "channel_"+std::to_string(i);
        int j = dstGrid->findBlindData(name.c_str());
        if (j<0) throw std::runtime_error("missing " + name);
        auto *metaData = this->dstMeta(j);// partially set in processGrid
        metaData->mDataClass = GridBlindDataClass::ChannelArray;
        metaData->mDataType  = mapToGridType<SrcValueT>();
        SrcValueT *blindData = const_cast<SrcValueT*>(metaData->template getBlindData<SrcValueT>());
        if (i>0) {// concurrent copy from previous channel
            nanovdb::forEach(0,valueCount,1024,[&](const nanovdb::Range1D &r){
                SrcValueT *dst=blindData+r.begin(), *end=dst+r.size(), *src=dst-valueCount;
                while(dst!=end) *dst++ = *src++;
            });
        } else {
            this->template copyValues<DstBuildT>(blindData);
        }
    }// loop over channels
    gridStats(*(this->template dstGrid<DstBuildT>()), std::min(StatsMode::BBox, mStats));
    updateChecksum(*(this->template dstGrid<DstBuildT>()), mChecksum);
}// CreateNanoGrid::postProcess<ValueIndex or ValueOnIndex>

//================================================================================================

template <typename SrcGridT>
template <typename DstBuildT>
typename enable_if<BuildTraits<DstBuildT>::is_index>::type
CreateNanoGrid<SrcGridT>::copyValues(SrcValueT *buffer)
{// copy values from the source grid into the provided buffer
    assert(mBufferPtr && buffer);
    using StatsT = typename FloatTraits<SrcValueT>::FloatType;

    if (this->valueCount()==0) this->template countValues<DstBuildT>();

    auto copyNodeValues = [&](const auto &node, SrcValueT *v) {
        if constexpr(BuildTraits<DstBuildT>::is_onindex) {
            for (auto it = node.cbeginValueOn(); it; ++it) *v++ = *it;
        } else {
            for (auto it = node.cbeginValueAll(); it; ++it) *v++ = *it;
        }
        if (mIncludeStats) {
            if constexpr(SrcNodeAccT::IS_NANOVDB) {// resolved at compile time
                *v++ = node.minimum();
                *v++ = node.maximum();
                if constexpr(is_same<SrcValueT, StatsT>::value) {
                    *v++ = node.average();
                    *v++ = node.stdDeviation();
                } else {// eg when SrcValueT=Vec3f and StatsT=float
                    *v++ = SrcValueT(node.average());
                    *v++ = SrcValueT(node.stdDeviation());
                }
            } else {// openvdb and nanovdb::build::Grid have no stats
                *v++ = buffer[0];// background
                *v++ = buffer[0];// background
                *v++ = buffer[0];// background
                *v++ = buffer[0];// background
            }
        }
    };// copyNodeValues

    const SrcRootT &root = mSrcNodeAcc.root();
    buffer[0] = root.background();// Value array always starts with the background value
    if (mIncludeTiles) {
        copyNodeValues(root, buffer + 1u);
        forEach(0, mSrcNodeAcc.nodeCount(2), 1, [&](const Range1D& r) {
            for (auto i = r.begin(); i!=r.end(); ++i) {
                copyNodeValues(mSrcNodeAcc.template node<2>(i), buffer + mValIdx[2][i]);
            }
        });
        forEach(0, mSrcNodeAcc.nodeCount(1), 1, [&](const Range1D& r) {
            for (auto i = r.begin(); i!=r.end(); ++i) {
                copyNodeValues(mSrcNodeAcc.template node<1>(i), buffer + mValIdx[1][i]);
            }
        });
    }
    forEach(0, mSrcNodeAcc.nodeCount(0), 4, [&](const Range1D& r) {
        for (auto i = r.begin(); i!=r.end(); ++i) {
            copyNodeValues(mSrcNodeAcc.template node<0>(i), buffer + mValIdx[0][i]);
        }
    });
}// CreateNanoGrid::copyValues<ValueIndex or ValueOnIndex>


//================================================================================================

#if defined(NANOVDB_USE_OPENVDB)

template <typename SrcGridT>
template<typename T>
typename disable_if<is_same<T, openvdb::tools::PointIndexGrid>::value ||
                    is_same<T, openvdb::points::PointDataGrid>::value, uint64_t>::type
CreateNanoGrid<SrcGridT>::countPoints() const
{
    static_assert(is_same<T, SrcGridT>::value, "expected default template parameter");
    return 0u;
}// CreateNanoGrid::countPoints<T>

template <typename SrcGridT>
template<typename T>
typename enable_if<is_same<T, openvdb::tools::PointIndexGrid>::value ||
                   is_same<T, openvdb::points::PointDataGrid>::value, uint64_t>::type
CreateNanoGrid<SrcGridT>::countPoints() const
{
    static_assert(is_same<T, SrcGridT>::value, "expected default template parameter");
    return reduce(0, mSrcNodeAcc.nodeCount(0), 8, uint64_t(0), [&](auto &r, uint64_t sum) {
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mSrcNodeAcc.template node<0>(i).getLastValue();
        return sum;}, std::plus<uint64_t>());
}// CreateNanoGrid::countPoints<PointIndexGrid or PointDataGrid>

template <typename SrcGridT>
template<typename DstBuildT, typename AttT, typename CodecT, typename T>
typename enable_if<is_same<openvdb::points::PointDataGrid, T>::value>::type
CreateNanoGrid<SrcGridT>::copyPointAttribute(size_t attIdx, AttT *attPtr)
{
    static_assert(std::is_same<SrcGridT, T>::value, "Expected default parameter");
    using HandleT = openvdb::points::AttributeHandle<AttT, CodecT>;
    forEach(0, mSrcNodeAcc.nodeCount(0), 16, [&](const auto& r) {
        auto *dstData = this->template dstNode<DstBuildT,0>(r.begin())->data();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstData) {
            auto& srcLeaf = mSrcNodeAcc.template node<0>(i);
            HandleT handle(srcLeaf.constAttributeArray(attIdx));
            AttT *p = attPtr + dstData->mMinimum;
            for (auto iter = srcLeaf.beginIndexOn(); iter; ++iter) *p++ = handle.get(*iter);
        }
    });
}// CreateNanoGrid::copyPointAttribute

#endif

//================================================================================================

template<typename SrcGridT, typename DstBuildT, typename BufferT>
typename disable_if<BuildTraits<DstBuildT>::is_index || BuildTraits<DstBuildT>::is_Fp, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode,
               ChecksumMode cMode,
               int verbose,
               const BufferT &buffer)
{
    CreateNanoGrid<SrcGridT> converter(srcGrid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.setVerbose(verbose);
    return converter.template getHandle<DstBuildT, BufferT>(buffer);
}// createNanoGrid<T>

//================================================================================================

template<typename SrcGridT, typename DstBuildT, typename BufferT>
typename enable_if<BuildTraits<DstBuildT>::is_index, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               uint32_t channels,
               bool includeStats,
               bool includeTiles,
               int verbose,
               const BufferT &buffer)
{
    CreateNanoGrid<SrcGridT> converter(srcGrid);
    converter.setVerbose(verbose);
    return converter.template getHandle<DstBuildT, BufferT>(channels, includeStats, includeTiles, buffer);
}

//================================================================================================

template<typename SrcGridT, typename DstBuildT, typename OracleT, typename BufferT>
typename enable_if<is_same<FpN, DstBuildT>::value, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode,
               ChecksumMode cMode,
               bool ditherOn,
               int verbose,
               const OracleT &oracle,
               const BufferT &buffer)
{
    CreateNanoGrid<SrcGridT> converter(srcGrid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    converter.setVerbose(verbose);
    return converter.template getHandle<DstBuildT, OracleT, BufferT>(oracle, buffer);
}// createNanoGrid<FpN>

//================================================================================================

template<typename SrcGridT, typename DstBuildT, typename BufferT>
typename enable_if<BuildTraits<DstBuildT>::is_FpX, GridHandle<BufferT>>::type
createNanoGrid(const SrcGridT &srcGrid,
               StatsMode sMode,
               ChecksumMode cMode,
               bool ditherOn,
               int verbose,
               const BufferT &buffer)
{
    CreateNanoGrid<SrcGridT> converter(srcGrid);
    converter.setStats(sMode);
    converter.setChecksum(cMode);
    converter.enableDithering(ditherOn);
    converter.setVerbose(verbose);
    return converter.template getHandle<DstBuildT, BufferT>(buffer);
}// createNanoGrid<Fp4,8,16>

//================================================================================================

#if defined(NANOVDB_USE_OPENVDB)
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
    using openvdb_UInt32Grid = openvdb::Grid<openvdb::UInt32Tree>;

    if (auto grid = openvdb::GridBase::grid<openvdb::FloatGrid>(base)) {
        return createNanoGrid<openvdb::FloatGrid, float, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::DoubleGrid>(base)) {
        return createNanoGrid<openvdb::DoubleGrid, double, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int32Grid>(base)) {
        return createNanoGrid<openvdb::Int32Grid, int32_t,BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int64Grid>(base)) {
        return createNanoGrid<openvdb::Int64Grid, int64_t, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb_UInt32Grid>(base)) {
        return createNanoGrid<openvdb_UInt32Grid, uint32_t, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3fGrid>(base)) {
        return createNanoGrid<openvdb::Vec3fGrid, nanovdb::Vec3f, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3dGrid>(base)) {
        return createNanoGrid<openvdb::Vec3dGrid, nanovdb::Vec3d, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::tools::PointIndexGrid>(base)) {
        return createNanoGrid<openvdb::tools::PointIndexGrid, uint32_t, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::points::PointDataGrid>(base)) {
        return createNanoGrid<openvdb::points::PointDataGrid, uint32_t, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::MaskGrid>(base)) {
        return createNanoGrid<openvdb::MaskGrid, nanovdb::ValueMask, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::BoolGrid>(base)) {
        return createNanoGrid<openvdb::BoolGrid, bool, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb_Vec4fGrid>(base)) {
        return createNanoGrid<openvdb_Vec4fGrid, nanovdb::Vec4f, BufferT>(*grid, sMode, cMode, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb_Vec4dGrid>(base)) {
        return createNanoGrid<openvdb_Vec4dGrid, nanovdb::Vec4d, BufferT>(*grid, sMode, cMode, verbose);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unrecognized OpenVDB grid type");
    }
}// openToNanoVDB
#endif

} // namespace nanovdb

#endif // NANOVDB_CREATE_NANOGRID_H_HAS_BEEN_INCLUDED
