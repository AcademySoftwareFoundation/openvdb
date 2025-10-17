// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/NanoToOpenVDB.h

    \author Ken Museth

    \date May 6, 2020

    \brief This class will deserialize an NanoVDB grid into an OpenVDB grid.

    \todo Add support for PointIndexGrid and PointDataGrid
*/

#include <nanovdb/NanoVDB.h> // manages and streams the raw memory buffer of a NanoVDB grid.
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/ForEach.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/SignedFloodFill.h>

#ifndef NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED

namespace nanovdb {

namespace tools {

namespace trait {

template<typename T>
struct MapToOpen { using type = T; };// see below for template specializations

template <typename T>
using OpenTree = typename openvdb::tree::Tree4<typename trait::MapToOpen<T>::type, 5, 4, 3>::Type;

template <typename T>
using OpenGrid = openvdb::Grid< OpenTree<T> >;

}// trait namespace

/// @brief Free-standing function that de-serializes a typed NanoVDB grid into an OpenVDB Grid
/// @tparam NanoBuildT Template build type for the @c grid
/// @param grid NanoVDB grid to be converted to an OpenVDB grid
/// @return If NanoBuildT is an index type, e.g. ValueOnIndex, a shared pointer to an OpenVDB GridBase
///         is returned. Otherwise a shared pointer to an OpenVDB Grid of a matching type
///.        (trait::OpenGrid<NanoBuildT>::type::Ptr) is returned.
template<typename NanoBuildT>
auto nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid);

/// @brief Free-standing function that de-serializes an IndexGrid with a sidecar into an OpenVDB Grid
/// @tparam NanoValueT Template type of the side car data accompanying the index @c grid
/// @param grid IndexGrid into the side car data
/// @param sideCar Linear array of side car data
/// @param gridClass GridClass corresponding to the side car data
/// @param gridName Name of the output grid
/// @return Shared pointer to an OpenVDB Grid of a matching type (trait::OpenGrid<NanoValueT>::type::Ptr)
template<typename NanoBuildT, typename NanoValueT>
typename util::enable_if<BuildTraits<NanoBuildT>::is_index,
                         typename trait::OpenGrid<NanoValueT>::Ptr>::type
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid,
              const NanoValueT *sideCar,
              GridClass gridClass = GridClass::Unknown,
              const char *gridName = nullptr);

/// @brief Forward declaration of free-standing function that de-serializes a grid in a NanoVDB GridHandle into an OpenVDB GridBase
/// @tparam BufferT Template type of the buffer used to allocate the grid handle
/// @param handle Handle for the input grid to be converted
/// @param gridID ID of the grid in the GridHandle to be converted (defaults to first grid)
/// @return Shared pointer to an OpenVDB BaseGrid
/// @note If grid number @c gridID in @c handle has GridType IndexGrid, then the first suitable blind data is used as the side-car data
template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, uint32_t gridID = 0);

// ================================================================================================

namespace trait {

template<>
struct MapToOpen<ValueMask> {using type = openvdb::ValueMask;};

template<typename T>
struct MapToOpen<math::Vec3<T>>{using type = openvdb::math::Vec3<T>;};

template<typename T>
struct MapToOpen<math::Vec4<T>>{using type = openvdb::math::Vec4<T>;};

template<>
struct MapToOpen<Fp4> {using type = float;};

template<>
struct MapToOpen<Fp8> {using type = float;};

template<>
struct MapToOpen<Fp16> {using type = float;};

template<>
struct MapToOpen<FpN> {using type = float;};

// Windows appears to always template instantiate the return type in enable_if
#ifdef _WIN32
template<>
struct MapToOpen<ValueIndex> {using type = int;};// dummy
template<>
struct MapToOpen<ValueOnIndex> {using type = int;};// dummy
template<>
struct MapToOpen<ValueIndexMask> {using type = int;};// dummy
template<>
struct MapToOpen<ValueOnIndexMask> {using type = int;};// dummy
#endif

template<typename T, uint32_t LEVEL>
struct OpenNode;

// Partial template specialization of above Node struct
template<typename T>
struct OpenNode<T, 0> {using type = openvdb::tree::LeafNode<typename MapToOpen<T>::type, 3>;};

template<typename T>
struct OpenNode<T, 1> {using type = openvdb::tree::InternalNode<typename OpenNode<T, 0>::type, 4>;};

template<typename T>
struct OpenNode<T, 2> {using type = openvdb::tree::InternalNode<typename OpenNode<T, 1>::type, 5>;};

/// @brief Maps from nanovdb::GridClass to openvdb::GridClass
/// @param gridClass nanovdb::GridClass
/// @return openvdb::GridClass
static openvdb::GridClass toOpenGridClass(GridClass gridClass)
{
    switch (gridClass) {
    case GridClass::LevelSet:
        return openvdb::GRID_LEVEL_SET;
    case GridClass::FogVolume:
        return openvdb::GRID_FOG_VOLUME;
    case GridClass::Staggered:
        return openvdb::GRID_STAGGERED;
    case GridClass::PointIndex:
        throw std::runtime_error("NanoToOpenVDB does not yet support PointIndexGrids");
    case GridClass::PointData:
        throw std::runtime_error("NanoToOpenVDB does not yet support PointDataGrids");
    default:
        return openvdb::GRID_UNKNOWN;
    }
}

template<typename T>
static const auto* mapPtr(const T *v) {return reinterpret_cast<const typename MapToOpen<T>::type*>(v);}

static openvdb::Coord mapCoord(const Coord &ijk){return openvdb::Coord(ijk[0], ijk[1], ijk[2]);}
template<uint32_t LOG2DIM>

static auto mapMask(const Mask<LOG2DIM> &m){return reinterpret_cast<const openvdb::util::NodeMask<LOG2DIM>&>(m);}

}// trait namespace

/// @brief This class will serialize an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
class NanoToOpenVDB
{
public:

    /// @brief Default c-tor
    NanoToOpenVDB(){}

    /// @brief Converts nanovdb::Grid<T> -> openvdb::Grid<T>::Ptr
    /// @tparam NanoBuildT Template type for the input NanoVDB grid
    /// @param grid NanoVDB grid to be converted
    /// @return Shared pointer to an OpenVDB grid of matching type
    template<typename NanoBuildT>
    typename util::enable_if<!BuildTraits<NanoBuildT>::is_index,
                             typename trait::OpenGrid<NanoBuildT>::Ptr>::type
    operator()(const NanoGrid<NanoBuildT>& grid);

    /// @brief Converts nanovdb::Grid<Index> + blind data  -> openvdb::GridBase::Ptr
    /// @param idxGrid NanoVDB IndexGrid with blind data to be converted
    /// @param blindDataID Id of the bind data to be used as the side-car. A negative values means
    ///        pick the first available (relevant) blind data.
    /// @return shared pointer to an OpenVDB GridBase
    template<typename NanoIndexT>
    typename util::enable_if<BuildTraits<NanoIndexT>::is_index,
                             openvdb::GridBase::Ptr>::type
    operator()(const NanoGrid<NanoIndexT>& idxGrid, int blindDataID = -1);

    /// @brief Converts nanovdb::Grid<NanoIndexT> + NanoValueT[]  -> openvdb::Grid<NanoValueT>::Ptr
    /// @tparam NanoValueT Template to of the side car data
    /// @param grid NanoVDB IndexGrid
    /// @param sideCar Linear array of side car data
    /// @param gridClass GridClass of the @c sideCar data
    /// @param name Name of the output grid
    /// @return Shared pointer to an OpenVDB grid of matching type
    template<typename NanoIndexT, typename NanoValueT>
    typename util::enable_if<BuildTraits<NanoIndexT>::is_index,
                             typename trait::OpenGrid<NanoValueT>::Ptr>::type
    operator()(const NanoGrid<NanoIndexT>& grid,
               const NanoValueT *sideCar,
               GridClass gridClass = GridClass::Unknown,
               const char *name = nullptr);

private:

    template<int LEVEL, typename NanoBuildT>
    typename util::enable_if<!BuildTraits<NanoBuildT>::is_index && (LEVEL == 1 || LEVEL == 2)>::type
    process(typename trait::OpenNode<NanoBuildT, LEVEL>::type*, const typename NanoNode<NanoBuildT, LEVEL>::type*);

    template<int LEVEL, typename NanoIndexT, typename NanoValueT>
    typename util::enable_if<BuildTraits<NanoIndexT>::is_index && (LEVEL == 1 || LEVEL == 2)>::type
    process(typename trait::OpenNode<NanoValueT, LEVEL>::type*, const typename NanoNode<NanoIndexT, LEVEL>::type*, const NanoValueT*);

    template<int LEVEL, typename NanoBuildT>
    typename util::enable_if<!BuildTraits<NanoBuildT>::is_index && LEVEL == 0>::type
    process(typename trait::OpenNode<NanoBuildT, LEVEL>::type*, const typename NanoNode<NanoBuildT, LEVEL>::type*);

    template<int LEVEL, typename NanoIndexT, typename NanoValueT>
    typename util::enable_if<BuildTraits<NanoIndexT>::is_index && LEVEL == 0>::type
    process(typename trait::OpenNode<NanoValueT, LEVEL>::type*, const typename NanoNode<NanoIndexT, LEVEL>::type*, const NanoValueT*);

}; // NanoToOpenVDB class

// ================================================================================================

template<typename NanoBuildT>
typename util::enable_if<!BuildTraits<NanoBuildT>::is_index,
                         typename trait::OpenGrid<NanoBuildT>::Ptr>::type
NanoToOpenVDB::operator()(const NanoGrid<NanoBuildT>& srcGrid)
{
    // Create an empty OpenVDB destination grid
    auto dstGrid = openvdb::createGrid<trait::OpenGrid<NanoBuildT>>(*trait::mapPtr(&srcGrid.tree().background()));
    dstGrid->setName(srcGrid.gridName()); // set grid name
    dstGrid->setGridClass(trait::toOpenGridClass(srcGrid.gridClass()));

    // set world to index transform
    const Map& nanoMap = reinterpret_cast<const GridData&>(srcGrid).mMap;
    auto  mat = openvdb::math::Mat4<double>::identity();
    mat.setMat3(openvdb::math::Mat3<double>(nanoMap.mMatD));
    mat = mat.transpose(); // the 3x3 in nanovdb is transposed relative to openvdb's 3x3
    mat.setTranslation(openvdb::math::Vec3<double>(nanoMap.mVecD));
    dstGrid->setTransform(openvdb::math::Transform::createLinearTransform(mat)); // calls simplify!

    // process root node and recursively call its child inner nodes
    auto &root = dstGrid->tree().root();
    auto *data = srcGrid.tree().root().data();
    for (uint32_t i=0; i<data->mTableSize; ++i) {
        auto *tile = data->tile(i);
        if (tile->isChild()) {
            auto *dstChild = new typename trait::OpenNode<NanoBuildT, 2>::type();// un-initialized
            this->template process<2, NanoBuildT>( dstChild, data->getChild(tile) );
            root.addChild( dstChild );
        } else {
            root.addTile(trait::mapCoord(tile->origin()), *trait::mapPtr(&tile->value), tile->state);
        }
    }

    return dstGrid;
}// NanoToOpenVDB::operator()(const NanoGrid<NanoBuildT>& grid)

template<typename NanoIndexT, typename NanoValueT>
typename util::enable_if<BuildTraits<NanoIndexT>::is_index,
                         typename trait::OpenGrid<NanoValueT>::Ptr>::type
NanoToOpenVDB::operator()(const NanoGrid<NanoIndexT>& indexGrid,
                          const NanoValueT *sideCar,
                          GridClass gridClass,
                          const char *name)
{
    // Create an empty OpenVDB destination grid
    auto dstGrid = openvdb::createGrid<trait::OpenGrid<NanoValueT>>(*trait::mapPtr(sideCar));//background is always the first value
    if (name) dstGrid->setName(name); // set grid name
    dstGrid->setGridClass(trait::toOpenGridClass(gridClass));

    // set world to index transform
    const Map& nanoMap = reinterpret_cast<const GridData&>(indexGrid).mMap;
    auto  mat = openvdb::math::Mat4<double>::identity();
    mat.setMat3(openvdb::math::Mat3<double>(nanoMap.mMatD));
    mat = mat.transpose(); // the 3x3 in nanovdb is transposed relative to openvdb's 3x3
    mat.setTranslation(openvdb::math::Vec3<double>(nanoMap.mVecD));
    dstGrid->setTransform(openvdb::math::Transform::createLinearTransform(mat)); // calls simplify!

    // process root node and recursively call its child inner nodes
    auto &root = dstGrid->tree().root();
    auto *data = indexGrid.tree().root().data();
    for (uint32_t i=0; i<data->mTableSize; ++i) {
        auto *tile = data->tile(i);
        if (tile->isChild()) {
            auto *dstChild = new typename trait::OpenNode<NanoValueT, 2>::type();// un-initialized
            this->template process<2, NanoIndexT, NanoValueT>( dstChild, data->getChild(tile), sideCar );
            root.addChild( dstChild );
        } else {
            root.addTile(trait::mapCoord(tile->origin()), *trait::mapPtr(sideCar + tile->value), tile->state);
        }
    }

    if constexpr(std::is_same<NanoIndexT, ValueOnIndex>::value) {
        if (gridClass == GridClass::LevelSet) openvdb::tools::signedFloodFill(dstGrid->tree());
    }

    return dstGrid;
}// NanoToOpenVDB::operator()(const NanoGrid<ValueIndex>& grid, const NanoValueT*)

template<typename NanoIndexT>
typename util::enable_if<BuildTraits<NanoIndexT>::is_index, openvdb::GridBase::Ptr>::type
NanoToOpenVDB::operator()(const NanoGrid<NanoIndexT>& idxGrid, int blindDataID)
{
    for (uint32_t i=0; i<idxGrid.blindDataCount(); ++i) {
        if (blindDataID >= 0 && int(i) != blindDataID) continue;
        const auto &metaData = idxGrid.blindMetaData(i);
        if (metaData.mDataClass != GridBlindDataClass::ChannelArray ||
            idxGrid.valueCount() != metaData.mValueCount) continue;

        const GridClass gridClass = toGridClass(metaData.mSemantic);
        switch (metaData.mDataType){
        case GridType::Float:
            NANOVDB_ASSERT(metaData.mValueSize == sizeof(float));
            return (*this)(idxGrid, idxGrid.template getBlindData<float>(i), gridClass);
        case GridType::Double:
            NANOVDB_ASSERT(metaData.mValueSize == sizeof(double));
            return (*this)(idxGrid, idxGrid.template getBlindData<double>(i), gridClass);
        case GridType::Int32:
            NANOVDB_ASSERT(metaData.mValueSize == sizeof(int32_t));
            return (*this)(idxGrid, idxGrid.template getBlindData<int32_t>(i), gridClass);
        case GridType::Vec3f:
            NANOVDB_ASSERT(metaData.mValueSize == sizeof(Vec3f));
            return (*this)(idxGrid, idxGrid.template getBlindData<Vec3f>(i), gridClass);
        default:// required to avoid compiler warning
            break;
        }
    }
    OPENVDB_THROW(openvdb::RuntimeError, "No valid side car located in the blind data!");
}// NanoToOpenVDB::operator()(const NanoGrid<NanoIndexT>& idxGrid)

// ================================================================================================

template<int LEVEL, typename NanoBuildT>
typename util::enable_if<!BuildTraits<NanoBuildT>::is_index && (LEVEL == 1 || LEVEL == 2)>::type
NanoToOpenVDB::process(typename trait::OpenNode<NanoBuildT, LEVEL>::type *dstNode,
                       const typename NanoNode<NanoBuildT, LEVEL>::type   *srcNode)
{
    using DstNodeT = typename trait::OpenNode<NanoBuildT, LEVEL>::type;
    using SrcNodeT = typename NanoNode<NanoBuildT, LEVEL>::type;
    dstNode->setOrigin(trait::mapCoord(srcNode->origin()));
    const auto &childMask = trait::mapMask(srcNode->childMask());
    const auto &valueMask = trait::mapMask(srcNode->valueMask());
    const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getValueMask()) = valueMask;
    const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getChildMask()) = childMask;
    auto* dstTable = const_cast<typename DstNodeT::UnionType*>(dstNode->getTable());
    auto* srcData  = srcNode->data();
    std::vector<std::pair<uint32_t, const typename SrcNodeT::ChildNodeType*>> childNodes;
    const auto childCount = childMask.countOn();
    childNodes.reserve(childCount);
    for (uint32_t n = 0; n < DstNodeT::NUM_VALUES; ++n) {
        if (childMask.isOn(n)) {
            childNodes.emplace_back(n, srcData->getChild(n));
        } else {
            dstTable[n].setValue(*trait::mapPtr(&srcData->mTable[n].value));
        }
    }
    auto kernel = [&](const auto& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto &p = childNodes[i];
            auto* dstChild = new typename DstNodeT::ChildNodeType();// un-initialized
            this->template process<LEVEL-1, NanoBuildT>(dstChild, p.second);
            dstTable[p.first].setChild( dstChild );
        }
    };

#if 0
    kernel(Range1D(0, childCount));
#else
    util::forEach(0, childCount, 1, kernel);
#endif
} // NanoToOpenVDB::process(const SrcNodeT *srcNode, DstNodeT *dstNode)

template<int LEVEL, typename NanoIndexT, typename NanoValueT>
typename util::enable_if<BuildTraits<NanoIndexT>::is_index && (LEVEL == 1 || LEVEL == 2)>::type
NanoToOpenVDB::process(typename trait::OpenNode<NanoValueT, LEVEL>::type *dstNode,
                       const typename NanoNode<NanoIndexT, LEVEL>::type   *srcNode,
                       const NanoValueT* sideCar)
{
    using DstNodeT = typename trait::OpenNode<NanoValueT, LEVEL>::type;
    using SrcNodeT = typename NanoNode<NanoIndexT, LEVEL>::type;
    dstNode->setOrigin(trait::mapCoord(srcNode->origin()));
    const auto &childMask = trait::mapMask(srcNode->childMask());
    const auto &valueMask = trait::mapMask(srcNode->valueMask());
    const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getValueMask()) = valueMask;
    const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getChildMask()) = childMask;
    auto* dstTable = const_cast<typename DstNodeT::UnionType*>(dstNode->getTable());
    auto* srcData  = srcNode->data();
    std::vector<std::pair<uint32_t, const typename SrcNodeT::ChildNodeType*>> childNodes;
    const auto childCount = childMask.countOn();
    childNodes.reserve(childCount);
    for (uint32_t n = 0; n < DstNodeT::NUM_VALUES; ++n) {
        if (childMask.isOn(n)) {
            childNodes.emplace_back(n, srcData->getChild(n));
        } else {
            dstTable[n].setValue(*trait::mapPtr(sideCar + srcData->mTable[n].value));
        }
    }
    auto kernel = [&](const auto& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto &p = childNodes[i];
            auto *dstChild = new typename DstNodeT::ChildNodeType();// un-initialized
            this->template process<LEVEL-1, NanoIndexT, NanoValueT>(dstChild, p.second, sideCar);
            dstTable[p.first].setChild( dstChild );
        }
    };

#if 0
    kernel(Range1D(0, childCount));
#else
    util::forEach(0, childCount, 1, kernel);
#endif
} // NanoToOpenVDB::process(const SrcNodeT* srcNode, const ValueT* sideCar)

// ================================================================================================

template<int LEVEL, typename NanoBuildT>
typename util::enable_if<!BuildTraits<NanoBuildT>::is_index && LEVEL == 0>::type
NanoToOpenVDB::process(typename trait::OpenNode<NanoBuildT, LEVEL>::type *dstLeaf,
                       const typename NanoNode<NanoBuildT, LEVEL>::type   *srcLeaf)
{
    dstLeaf->setOrigin(trait::mapCoord(srcLeaf->origin()));
    dstLeaf->setValueMask(trait::mapMask(srcLeaf->valueMask()));

    if constexpr(!BuildTraits<NanoBuildT>::is_special) {
        const auto* src = trait::mapPtr(srcLeaf->data()->mValues);// doesn't work for compressed data, bool or ValueMask
        for (auto *dst = dstLeaf->buffer().data(), *end = dst + 512; dst != end; dst += 4, src += 4) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
        }
    } else if constexpr(BuildTraits<NanoBuildT>::is_Fp) {
        float *dst = dstLeaf->buffer().data();
        for (int i=0; i!=512; i+=4) {
            *dst++ = srcLeaf->getValue(i);
            *dst++ = srcLeaf->getValue(i+1);
            *dst++ = srcLeaf->getValue(i+2);
            *dst++ = srcLeaf->getValue(i+3);
        }
    } else if constexpr(std::is_same<NanoBuildT, bool>::value) {
        reinterpret_cast<openvdb::util::NodeMask<3>&>(dstLeaf->buffer()) = trait::mapMask(srcLeaf->data()->mValues);
    } else if constexpr(!std::is_same<NanoBuildT, ValueMask>::value) {
        OPENVDB_THROW(openvdb::RuntimeError, "Unsupported NanoBuildT in NanoToOpenVDB::process(NanoLeaf<NanoBuildT>)");
    }
}// NanoToOpenVDB::process(const NanoLeaf<NanoBuildT>* srcLeaf)

template<int LEVEL, typename NanoIndexT, typename NanoValueT>
typename util::enable_if<BuildTraits<NanoIndexT>::is_index && LEVEL == 0>::type
NanoToOpenVDB::process(typename trait::OpenNode<NanoValueT, LEVEL>::type *dstLeaf,
                       const typename NanoNode<NanoIndexT, LEVEL>::type   *srcLeaf,
                       const NanoValueT *sideCar)
{
    dstLeaf->setOrigin(trait::mapCoord(srcLeaf->origin()));
    dstLeaf->setValueMask(trait::mapMask(srcLeaf->valueMask()));

    if constexpr(!BuildTraits<NanoValueT>::is_special) {
        if constexpr(util::is_same<NanoIndexT, ValueIndex, ValueIndexMask>::value) {
            const auto* src = trait::mapPtr(sideCar + srcLeaf->data()->mOffset);
            for (auto *dst = dstLeaf->buffer().data(), *end = dst + 512; dst != end; dst += 4, src += 4) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        } else {
            const auto* src = trait::mapPtr(sideCar);
            auto *dst = dstLeaf->buffer().data();
            for (int i = 0; i < 512; i += 4) {
                *dst++ = src[srcLeaf->getValue(i  )];
                *dst++ = src[srcLeaf->getValue(i+1)];
                *dst++ = src[srcLeaf->getValue(i+2)];
                *dst++ = src[srcLeaf->getValue(i+3)];
            }
        }
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unsupported NanoValueT in NanoToOpenVDB::process(NanoLeaf<NanoIndexT>, NanoValueT*)");
    }
}// NanoToOpenVDB::process(const NanoLeaf<NanoIndexT>* srcLeaf, const NanoValueT *sideCar)

// ================================================================================================

template<typename NanoBuildT>
auto nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid)
{
    NanoToOpenVDB tmp;
    return tmp(grid);
}

template<typename NanoBuildT, typename NanoValueT>
typename util::enable_if<BuildTraits<NanoBuildT>::is_index,
                         typename trait::OpenGrid<NanoValueT>::Ptr>::type
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid,
              const NanoValueT *sideCar,
              GridClass gridClass,
              const char *gridName)
{
    NanoToOpenVDB tmp;
    return tmp(grid, sideCar, gridClass, gridName);
}

template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, uint32_t n)
{
    if (auto grid = handle.template grid<float>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<double>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<int32_t>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<int64_t>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<bool>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Fp4>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Fp8>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Fp16>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<FpN>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<ValueMask>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Vec3f>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Vec3d>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Vec4f>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<Vec4d>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<ValueOnIndex>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<ValueOnIndexMask>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<ValueIndex>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else if (auto grid = handle.template grid<ValueIndexMask>(n)) {
        return tools::nanoToOpenVDB(*grid);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unsupported NanoVDB grid type!");
    }
}// tools::nanoToOpenVDB

}// namespace tools

/// @brief Forward declaration of free-standing function that de-serializes a typed NanoVDB grid into an OpenVDB Grid
template<typename NanoBuildT>
[[deprecated("Use nanovdb::tools::nanoToOpenVDB instead.")]]
typename tools::trait::OpenGrid<NanoBuildT>::Ptr
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid)
{
    return tools::nanoToOpenVDB(grid);
}

/// @brief Forward declaration of free-standing function that de-serializes a NanoVDB GridHandle into an OpenVDB GridBase
template<typename BufferT>
[[deprecated("Use nanovdb::tools::nanoToOpenVDB instead.")]]
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, uint32_t n = 0)
{
    return tools::nanoToOpenVDB(handle, n);
}

} // namespace nanovdb

#endif // NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
