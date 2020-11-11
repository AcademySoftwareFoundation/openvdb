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
#include <openvdb/openvdb.h>

#include <tbb/parallel_for.h>

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

    const SrcGridT*    mSrcGrid;
    DstGridT*          mDstGrid;
    std::vector<void*> mDstNodes[3];

public:
    /// @brief Construction from an existing const OpenVDB Grid.
    NanoToOpenVDB(){};

    /// @brief Return a shared pointer to a NanoVDB grid constructed from the specified OpenVDB grid
    typename DstGridT::Ptr operator()(const NanoGrid<ValueType>& grid, int verbose = 0);

private:
    void processGrid();

    void processLeafs();

    template<typename SrcNodeT, typename DstNodeT>
    void processNodes();

}; // NanoToOpenVDB class

template<typename T>
struct ConvertTrait
{
    using Type = T;
};

template<typename T>
struct ConvertTrait<Vec3<T>>
{
    using Type = openvdb::math::Vec3<T>;
};

template<typename T>
typename NanoToOpenVDB<T>::DstGridT::Ptr
NanoToOpenVDB<T>::operator()(const NanoGrid<T>& grid, int /*verbose*/)
{
    // since the input nanovdb grid might use nanovdb types (Coord, Mask, Vec3) we case to use openvdb types
    mSrcGrid = reinterpret_cast<const SrcGridT*>(&grid);

    this->processGrid();

    this->processLeafs();

    this->template processNodes<SrcNode1, DstNode1>();

    this->template processNodes<SrcNode2, DstNode2>();

    auto& dstRoot = mDstGrid->tree().root();
    for (auto& child : mDstNodes[2])
        dstRoot.addChild(reinterpret_cast<DstNode2*>(child));

    return openvdb::SharedPtr<DstGridT>(mDstGrid);
}

template<typename T>
void NanoToOpenVDB<T>::processGrid()
{
    mDstGrid = new DstGridT(mSrcGrid->tree().background());
    mDstGrid->setName(mSrcGrid->gridName()); // set grid name
    switch (mSrcGrid->gridClass()) { // set grid class
    case nanovdb::GridClass::LevelSet:
        mDstGrid->setGridClass(openvdb::GRID_LEVEL_SET);
        break;
    case nanovdb::GridClass::FogVolume:
        mDstGrid->setGridClass(openvdb::GRID_FOG_VOLUME);
        break;
    case nanovdb::GridClass::Staggered:
        mDstGrid->setGridClass(openvdb::GRID_STAGGERED);
        break;
    default:
        mDstGrid->setGridClass(openvdb::GRID_UNKNOWN);
    }
    // set transform
    const nanovdb::Map& nanoMap = reinterpret_cast<const GridData*>(mSrcGrid)->mMap;
    auto                mat = openvdb::math::Mat4<double>::identity();
    mat.setMat3(openvdb::math::Mat3<double>(nanoMap.mMatD));
    mat.transpose(); // the 3x3 in nanovdb is transposed relative to openvdb's 3x3
    mat.setTranslation(openvdb::math::Vec3<double>(nanoMap.mVecD));
    mDstGrid->setTransform(openvdb::math::Transform::createLinearTransform(mat)); // calls simplify!
}

template<typename T>
void NanoToOpenVDB<T>::processLeafs()
{
    const SrcTreeT& srcTree = mSrcGrid->tree();
    mDstNodes[0].resize(srcTree.template nodeCount<SrcNode0>());

    auto kernel = [&](const tbb::blocked_range<size_t>& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            const SrcNode0* srcNode = srcTree.template getNode<SrcNode0>(i);
            DstNode0*       dstNode = new DstNode0(); // un-initialized for fast construction
            dstNode->setOrigin(srcNode->origin());
            dstNode->setValueMask(srcNode->valueMask());
            const ValueT* src = &srcNode->getValue(0);
            for (ValueT *dst = dstNode->buffer().data(), *end = dst + DstNode0::SIZE; dst != end; dst += 4, src += 4) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
            mDstNodes[0][i] = dstNode;
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mDstNodes[0].size(), 4), kernel);

} // processLeafs

template<typename T>
template<typename SrcNodeT, typename DstNodeT>
void NanoToOpenVDB<T>::processNodes()
{
    const SrcTreeT& srcTree = mSrcGrid->tree();
    mDstNodes[DstNodeT::LEVEL].resize(srcTree.template nodeCount<SrcNodeT>());

    auto kernel = [&](const tbb::blocked_range<size_t>& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            const SrcNodeT* srcNode = srcTree.template getNode<SrcNodeT>(i);
            DstNodeT*       dstNode = new DstNodeT(); // un-initialized for fast construction
            dstNode->setOrigin(srcNode->origin());
            const auto& childMask = srcNode->childMask();
            const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getValueMask()) = srcNode->valueMask();
            const_cast<typename DstNodeT::NodeMaskType&>(dstNode->getChildMask()) = childMask;
            auto* dstTable = const_cast<typename DstNodeT::UnionType*>(dstNode->getTable());
            auto* srcTable = reinterpret_cast<const typename SrcNodeT::DataType*>(srcNode)->mTable;
            for (uint32_t n = 0; n < DstNodeT::NUM_VALUES; ++n) {
                if (childMask.isOn(n)) {
                    void* child = mDstNodes[DstNodeT::LEVEL - 1][srcTable[n].childID];
                    dstTable[n].setChild(reinterpret_cast<typename DstNodeT::ChildNodeType*>(child));
                } else {
                    dstTable[n].setValue(srcTable[n].value);
                }
            }
            mDstNodes[DstNodeT::LEVEL][i] = dstNode;
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mDstNodes[DstNodeT::LEVEL].size(), 4), kernel);
} // processNodes

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
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unrecognized NanoVDB grid type");
    }
}

} // namespace nanovdb

#endif // NANOVDB_NANOTOOPENVDB_H_HAS_BEEN_INCLUDED
