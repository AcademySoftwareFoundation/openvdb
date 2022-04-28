// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file NanoToOpenVDB.h

    \author Ken Museth

    \date May 6, 2020

    \brief This class will deserialize an NanoVDB grid into an OpenVDB grid.

    \todo Add support for PointIndexGrid and PointDataGrid
*/

#include <nanovdb/NanoVDB.h> // manages and streams the raw memory buffer of a NanoVDB grid.
#include <nanovdb/util/GridHandle.h>
#include "ForEach.h"

#include <openvdb/openvdb.h>

#ifndef NANOVDB_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED

namespace nanovdb {

template<typename T>
struct ConvertTrait;

/// @brief Forward declaration of free-standing function that de-serializes a typed NanoVDB grid into an OpenVDB Grid
template<typename ValueT>
typename openvdb::Grid<typename openvdb::tree::Tree4<typename ConvertTrait<ValueT>::Type>::Type>::Ptr
nanoToOpenVDB(const NanoGrid<ValueT>& grid, int verbose = 0);

/// @brief Forward declaration of free-standing function that de-serializes a NanoVDB GridHandle into an OpenVDB GridBase
template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, int verbose = 0);

/// @brief This class will serialize an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
template<typename ValueType>
class NanoToOpenVDB
{
    using ValueT = typename ConvertTrait<ValueType>::Type; // e.g. float -> float but nanovdb::Vec3<float> -> openvdb::Vec3<float>
    using SrcNode0 = LeafNode<ValueT, openvdb::Coord, openvdb::util::NodeMask>; // note that it's using openvdb types!
    using SrcNode1 = InternalNode<SrcNode0>;
    using SrcNode2 = InternalNode<SrcNode1>;
    using SrcRootT = RootNode<SrcNode2>;
    using SrcTreeT = Tree<SrcRootT>;
    using SrcGridT = Grid<SrcTreeT>;

    using DstNode0 = openvdb::tree::LeafNode<ValueT, SrcNode0::LOG2DIM>; // leaf
    using DstNode1 = openvdb::tree::InternalNode<DstNode0, SrcNode1::LOG2DIM>; // lower
    using DstNode2 = openvdb::tree::InternalNode<DstNode1, SrcNode2::LOG2DIM>; // upper
    using DstRootT = openvdb::tree::RootNode<DstNode2>;
    using DstTreeT = openvdb::tree::Tree<DstRootT>;
    using DstGridT = openvdb::Grid<DstTreeT>;

public:
    /// @brief Construction from an existing const OpenVDB Grid.
    NanoToOpenVDB(){};

    /// @brief Return a shared pointer to a NanoVDB grid constructed from the specified OpenVDB grid
    typename DstGridT::Ptr operator()(const NanoGrid<ValueType>& grid, int verbose = 0);

private:

    template<typename SrcNodeT, typename DstNodeT>
    DstNodeT* processNode(const SrcNodeT*);

    DstNode2* process(const SrcNode2* node) {return this->template processNode<SrcNode2, DstNode2>(node);}
    DstNode1* process(const SrcNode1* node) {return this->template processNode<SrcNode1, DstNode1>(node);}
    DstNode0* process(const SrcNode0* node);
}; // NanoToOpenVDB class

template<typename T>
struct ConvertTrait
{
    using Type = T;
};

template<typename T>
struct ConvertTrait< Vec3<T> >
{
    using Type = openvdb::math::Vec3<T>;
};

template<typename T>
struct ConvertTrait< Vec4<T> >
{
    using Type = openvdb::math::Vec4<T>;
};

template<typename T>
typename NanoToOpenVDB<T>::DstGridT::Ptr
NanoToOpenVDB<T>::operator()(const NanoGrid<T>& grid, int /*verbose*/)
{
    // since the input nanovdb grid might use nanovdb types (Coord, Mask, Vec3) we cast to use openvdb types
    const SrcGridT *srcGrid = reinterpret_cast<const SrcGridT*>(&grid);
    auto dstGrid = openvdb::createGrid<DstGridT>(srcGrid->tree().background());
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
    case nanovdb::GridClass::Topology:
        throw std::runtime_error("NanoToOpenVDB does not yet support Mask (or Topology) Grids");
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
            root.addTile(tile->origin(), tile->value, tile->state);
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
            dstTable[n].setValue(srcData->mTable[n].value);
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
    forEach(0, childCount, 1, kernel);
#endif
    return dstNode;
} // processNode

template<typename T>
typename NanoToOpenVDB<T>::DstNode0*
NanoToOpenVDB<T>::process(const SrcNode0 *srcNode)
{
    DstNode0* dstNode = new DstNode0(); // un-initialized for fast construction
    dstNode->setOrigin(srcNode->origin());
    dstNode->setValueMask(srcNode->valueMask());

    const ValueT* src = srcNode->data()->mValues;// doesn't work for compressed data, bool or ValueMask
    for (ValueT *dst = dstNode->buffer().data(), *end = dst + DstNode0::SIZE; dst != end; dst += 4, src += 4) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    }

    return dstNode;
} // process(SrcNode0)

template<typename ValueT>
typename openvdb::Grid<typename openvdb::tree::Tree4<typename ConvertTrait<ValueT>::Type>::Type>::Ptr
nanoToOpenVDB(const NanoGrid<ValueT>& grid, int verbose)
{
    nanovdb::NanoToOpenVDB<ValueT> tmp;
    return tmp(grid, verbose);
}

template<typename BufferT>
openvdb::GridBase::Ptr
nanoToOpenVDB(const GridHandle<BufferT>& handle, int verbose)
{
    if (auto grid = handle.template grid<float>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<double>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<int32_t>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<int64_t>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec3f>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec3d>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec4f>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else if (auto grid = handle.template grid<nanovdb::Vec4d>()) {
        return nanovdb::nanoToOpenVDB(*grid, verbose);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unsupported NanoVDB grid type");
    }
}

} // namespace nanovdb

#endif // NANOVDB_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
