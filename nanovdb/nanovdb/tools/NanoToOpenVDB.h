// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

#ifndef NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED

template<typename T>
struct ConvertTrait {using Type = T;};

template<typename T>
struct ConvertTrait<nanovdb::math::Vec3<T>> {using Type = openvdb::math::Vec3<T>;};

template<typename T>
struct ConvertTrait<nanovdb::math::Vec4<T>> {using Type = openvdb::math::Vec4<T>;};

template<>
struct ConvertTrait<nanovdb::Fp4> {using Type = float;};

template<>
struct ConvertTrait<nanovdb::Fp8> {using Type = float;};

template<>
struct ConvertTrait<nanovdb::Fp16> {using Type = float;};

template<>
struct ConvertTrait<nanovdb::FpN> {using Type = float;};

template<>
struct ConvertTrait<nanovdb::ValueMask> {using Type = openvdb::ValueMask;};

namespace nanovdb {

namespace tools {

/// @brief Forward declaration of free-standing function that de-serializes a typed NanoVDB grid into an OpenVDB Grid
template<typename NanoBuildT>
typename openvdb::Grid<typename openvdb::tree::Tree4<typename ConvertTrait<NanoBuildT>::Type>::Type>::Ptr
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid, int verbose = 0);

/// @brief Forward declaration of free-standing function that de-serializes a NanoVDB GridHandle into an OpenVDB GridBase
template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, int verbose = 0, uint32_t n = 0);

/// @brief This class will serialize an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
template<typename NanoBuildT>
class NanoToOpenVDB
{
    using NanoNode0  = nanovdb::LeafNode<NanoBuildT, openvdb::Coord, openvdb::util::NodeMask>; // note that it's using openvdb coord nd mask types!
    using NanoNode1  = nanovdb::InternalNode<NanoNode0>;
    using NanoNode2  = nanovdb::InternalNode<NanoNode1>;
    using NanoRootT  = nanovdb::RootNode<NanoNode2>;
    using NanoTreeT  = nanovdb::Tree<NanoRootT>;
    using NanoGridT  = nanovdb::Grid<NanoTreeT>;
    using NanoValueT = typename NanoGridT::ValueType;

    using OpenBuildT = typename ConvertTrait<NanoBuildT>::Type; // e.g. float -> float but nanovdb::math::Vec3<float> -> openvdb::Vec3<float>
    using OpenNode0  = openvdb::tree::LeafNode<OpenBuildT, NanoNode0::LOG2DIM>; // leaf
    using OpenNode1  = openvdb::tree::InternalNode<OpenNode0, NanoNode1::LOG2DIM>; // lower
    using OpenNode2  = openvdb::tree::InternalNode<OpenNode1, NanoNode2::LOG2DIM>; // upper
    using OpenRootT  = openvdb::tree::RootNode<OpenNode2>;
    using OpenTreeT  = openvdb::tree::Tree<OpenRootT>;
    using OpenGridT  = openvdb::Grid<OpenTreeT>;
    using OpenValueT = typename OpenGridT::ValueType;

public:
    /// @brief Construction from an existing const OpenVDB Grid.
    NanoToOpenVDB(){};

    /// @brief Return a shared pointer to a NanoVDB grid constructed from the specified OpenVDB grid
    typename OpenGridT::Ptr operator()(const NanoGrid<NanoBuildT>& grid, int verbose = 0);

private:

    template<typename NanoNodeT, typename OpenNodeT>
    OpenNodeT* processNode(const NanoNodeT*);

    OpenNode2* process(const NanoNode2* node) {return this->template processNode<NanoNode2, OpenNode2>(node);}
    OpenNode1* process(const NanoNode1* node) {return this->template processNode<NanoNode1, OpenNode1>(node);}

    template <typename NanoLeafT>
    typename std::enable_if<!std::is_same<bool, typename NanoLeafT::BuildType>::value &&
                            !std::is_same<ValueMask, typename NanoLeafT::BuildType>::value &&
                            !std::is_same<Fp4, typename NanoLeafT::BuildType>::value &&
                            !std::is_same<Fp8, typename NanoLeafT::BuildType>::value &&
                            !std::is_same<Fp16,typename NanoLeafT::BuildType>::value &&
                            !std::is_same<FpN, typename NanoLeafT::BuildType>::value,
                            OpenNode0*>::type
    process(const NanoLeafT* node);

    template <typename NanoLeafT>
    typename std::enable_if<std::is_same<Fp4, typename NanoLeafT::BuildType>::value ||
                            std::is_same<Fp8, typename NanoLeafT::BuildType>::value ||
                            std::is_same<Fp16,typename NanoLeafT::BuildType>::value ||
                            std::is_same<FpN, typename NanoLeafT::BuildType>::value,
                            OpenNode0*>::type
    process(const NanoLeafT* node);

    template <typename NanoLeafT>
    typename std::enable_if<std::is_same<ValueMask, typename NanoLeafT::BuildType>::value,
                            OpenNode0*>::type
    process(const NanoLeafT* node);

    template <typename NanoLeafT>
    typename std::enable_if<std::is_same<bool, typename NanoLeafT::BuildType>::value,
                            OpenNode0*>::type
    process(const NanoLeafT* node);

    /// converts nanovdb value types to openvdb value types, e.g. nanovdb::Vec3f& -> openvdb::Vec3f&
    static const OpenValueT& Convert(const NanoValueT &v) {return reinterpret_cast<const OpenValueT&>(v);}
    static const OpenValueT* Convert(const NanoValueT *v) {return reinterpret_cast<const OpenValueT*>(v);}

}; // NanoToOpenVDB class

template<typename NanoBuildT>
typename NanoToOpenVDB<NanoBuildT>::OpenGridT::Ptr
NanoToOpenVDB<NanoBuildT>::operator()(const NanoGrid<NanoBuildT>& grid, int /*verbose*/)
{
    // since the input nanovdb grid might use nanovdb types (Coord, Mask, Vec3) we cast to use openvdb types
    const NanoGridT *srcGrid = reinterpret_cast<const NanoGridT*>(&grid);

    auto dstGrid = openvdb::createGrid<OpenGridT>(Convert(srcGrid->tree().background()));
    dstGrid->setName(srcGrid->gridName()); // set grid name
    switch (srcGrid->gridClass()) { // set grid class
    case nanovdb::GridClass::LevelSet:
        dstGrid->setGridClass(openvdb::GRID_LEVEL_SET);
        break;
    case nanovdb::GridClass::FogVolume:
        dstGrid->setGridClass(openvdb::GRID_FOG_VOLUME);
        break;
    case nanovdb::GridClass::Staggered:
        dstGrid->setGridClass(openvdb::GRID_STAGGERED);
        break;
    case nanovdb::GridClass::PointIndex:
        throw std::runtime_error("NanoToOpenVDB does not yet support PointIndexGrids");
    case nanovdb::GridClass::PointData:
        throw std::runtime_error("NanoToOpenVDB does not yet support PointDataGrids");
    default:
        dstGrid->setGridClass(openvdb::GRID_UNKNOWN);
    }
    // set transform
    const nanovdb::Map& nanoMap = reinterpret_cast<const GridData*>(srcGrid)->mMap;
    auto                mat = openvdb::math::Mat4<double>::identity();
    mat.setMat3(openvdb::math::Mat3<double>(nanoMap.mMatD));
    mat.transpose(); // the 3x3 in nanovdb is transposed relative to openvdb's 3x3
    mat.setTranslation(openvdb::math::Vec3<double>(nanoMap.mVecD));
    dstGrid->setTransform(openvdb::math::Transform::createLinearTransform(mat)); // calls simplify!

    // process root node
    auto &root = dstGrid->tree().root();
    auto *data = srcGrid->tree().root().data();
    for (uint32_t i=0; i<data->mTableSize; ++i) {
        auto *tile = data->tile(i);
        if (tile->isChild()) {
            root.addChild( this->process( data->getChild(tile)) );
        } else {
            root.addTile(tile->origin(), Convert(tile->value), tile->state);
        }
    }

    return dstGrid;
}

template<typename T>
template<typename SrcNodeT, typename DstNodeT>
DstNodeT*
NanoToOpenVDB<T>::processNode(const SrcNodeT *srcNode)
{
    DstNodeT *dstNode = new DstNodeT(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    const auto& childMask = srcNode->childMask();
    const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getValueMask()) = srcNode->valueMask();
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
            dstTable[n].setValue(Convert(srcData->mTable[n].value));
        }
    }
    auto kernel = [&](const auto& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto &p = childNodes[i];
            dstTable[p.first].setChild( this->process(p.second) );
        }
    };

#if 0
    kernel(Range1D(0, childCount));
#else
    util::forEach(0, childCount, 1, kernel);
#endif
    return dstNode;
} // processNode

template<typename T>
template <typename NanoLeafT>
inline typename std::enable_if<!std::is_same<bool, typename NanoLeafT::BuildType>::value &&
                               !std::is_same<ValueMask, typename NanoLeafT::BuildType>::value &&
                               !std::is_same<Fp4, typename NanoLeafT::BuildType>::value &&
                               !std::is_same<Fp8, typename NanoLeafT::BuildType>::value &&
                               !std::is_same<Fp16,typename NanoLeafT::BuildType>::value &&
                               !std::is_same<FpN, typename NanoLeafT::BuildType>::value,
                               typename NanoToOpenVDB<T>::OpenNode0*>::type
NanoToOpenVDB<T>::process(const NanoLeafT *srcNode)
{
    static_assert(std::is_same<NanoLeafT, NanoNode0>::value, "NanoToOpenVDB<FpN>::process assert failed");
    OpenNode0* dstNode = new OpenNode0(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    dstNode->setValueMask(srcNode->valueMask());

    const auto* src = Convert(srcNode->data()->mValues);// doesn't work for compressed data, bool or ValueMask
    for (auto *dst = dstNode->buffer().data(), *end = dst + OpenNode0::SIZE; dst != end; dst += 4, src += 4) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    }

    return dstNode;
} // process(NanoNode0)

template<typename T>
template <typename NanoLeafT>
inline typename std::enable_if<std::is_same<Fp4, typename NanoLeafT::BuildType>::value ||
                               std::is_same<Fp8, typename NanoLeafT::BuildType>::value ||
                               std::is_same<Fp16,typename NanoLeafT::BuildType>::value ||
                               std::is_same<FpN, typename NanoLeafT::BuildType>::value,
                               typename NanoToOpenVDB<T>::OpenNode0*>::type
NanoToOpenVDB<T>::process(const NanoLeafT *srcNode)
{
    static_assert(std::is_same<NanoLeafT, NanoNode0>::value, "NanoToOpenVDB<T>::process assert failed");
    OpenNode0* dstNode = new OpenNode0(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    dstNode->setValueMask(srcNode->valueMask());
    float *dst = dstNode->buffer().data();
    for (int i=0; i!=512; i+=4) {
        *dst++ = srcNode->getValue(i);
        *dst++ = srcNode->getValue(i+1);
        *dst++ = srcNode->getValue(i+2);
        *dst++ = srcNode->getValue(i+3);
    }

    return dstNode;
} // process(NanoNode0)

template<typename T>
template <typename NanoLeafT>
inline typename std::enable_if<std::is_same<ValueMask, typename NanoLeafT::BuildType>::value,
                               typename NanoToOpenVDB<T>::OpenNode0*>::type
NanoToOpenVDB<T>::process(const NanoLeafT *srcNode)
{
    static_assert(std::is_same<NanoLeafT, NanoNode0>::value, "NanoToOpenVDB<ValueMask>::process assert failed");
    OpenNode0* dstNode = new OpenNode0(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    dstNode->setValueMask(srcNode->valueMask());

    return dstNode;
} // process(NanoNode0)

template<typename T>
template <typename NanoLeafT>
inline typename std::enable_if<std::is_same<bool, typename NanoLeafT::BuildType>::value,
                               typename NanoToOpenVDB<T>::OpenNode0*>::type
NanoToOpenVDB<T>::process(const NanoLeafT *srcNode)
{
    static_assert(std::is_same<NanoLeafT, NanoNode0>::value, "NanoToOpenVDB<ValueMask>::process assert failed");
    OpenNode0* dstNode = new OpenNode0(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    dstNode->setValueMask(srcNode->valueMask());
    reinterpret_cast<openvdb::util::NodeMask<3>&>(dstNode->buffer()) = srcNode->data()->mValues;

    return dstNode;
} // process(NanoNode0)

template<typename NanoBuildT>
inline typename openvdb::Grid<typename openvdb::tree::Tree4<typename ConvertTrait<NanoBuildT>::Type>::Type>::Ptr
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid, int verbose)
{
    NanoToOpenVDB<NanoBuildT> tmp;
    return tmp(grid, verbose);
}

template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, int verbose, uint32_t n)
{
    if (auto grid = handle.template grid<float>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<double>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<int32_t>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<int64_t>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<bool>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Fp4>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Fp8>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Fp16>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::FpN>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::ValueMask>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec3f>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec3d>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec4f>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec4d>(n)) {
        return tools::nanoToOpenVDB(*grid, verbose);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unsupported NanoVDB grid type!");
    }
}// tools::nanoToOpenVDB

}// namespace tools

/// @brief Forward declaration of free-standing function that de-serializes a typed NanoVDB grid into an OpenVDB Grid
template<typename NanoBuildT>
[[deprecated("Use nanovdb::tools::nanoToOpenVDB instead.")]]
typename openvdb::Grid<typename openvdb::tree::Tree4<typename ConvertTrait<NanoBuildT>::Type>::Type>::Ptr
nanoToOpenVDB(const NanoGrid<NanoBuildT>& grid, int verbose = 0)
{
    return tools::nanoToOpenVDB(grid, verbose);
}

/// @brief Forward declaration of free-standing function that de-serializes a NanoVDB GridHandle into an OpenVDB GridBase
template<typename BufferT>
[[deprecated("Use nanovdb::tools::nanoToOpenVDB instead.")]]
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, int verbose = 0, uint32_t n = 0)
{
    return tools::nanoToOpenVDB(handle, verbose, n);
}

} // namespace nanovdb

#endif // NANOVDB_TOOLS_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
