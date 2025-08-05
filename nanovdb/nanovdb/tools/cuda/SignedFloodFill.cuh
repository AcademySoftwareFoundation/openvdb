// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/SignedFloodFill.cuh

    \author Ken Museth

    \date May 3, 2023

    \brief Performs signed flood-fill operation on the hierarchical tree structure on the device

    \todo This tools needs to handle the (extremely) rare case when the root node
          needs to be modified during the signed flood fill operation. This happens
          when the root-table needs to be expanded with tile values (of size 4096^3)
          that are completely inside the implicit surface.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_SIGNEDFLOODFILL_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_SIGNEDFLOODFILL_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/io/IO.h>// debugging needs io of Coord

namespace nanovdb {

namespace tools::cuda {

/// @brief Performs signed flood-fill operation on the hierarchical tree structure on the device
/// @tparam BuildT Build type of the grid to be flood-filled
/// @param d_grid Non-const device pointer to the grid that will be flood-filled
/// @param verbose If true timing information will be printed to the terminal
/// @param stream optional cuda stream
template<typename BuildT>
typename util::enable_if<BuildTraits<BuildT>::is_float, void>::type
signedFloodFill(NanoGrid<BuildT> *d_grid, bool verbose = false, cudaStream_t stream = 0);

template<typename BuildT>
class SignedFloodFill
{
public:
    SignedFloodFill(bool verbose = false, cudaStream_t stream = 0)
        : mStream(stream), mVerbose(verbose) {}

    /// @brief Toggle on and off verbose mode
    /// @param on if true verbose is turned on
    void setVerbose(bool on = true) {mVerbose = on;}

    void operator()(NanoGrid<BuildT> *d_grid);

private:
    cudaStream_t      mStream{0};
    util::cuda::Timer mTimer;
    bool              mVerbose{false};

};// SignedFloodFill

//================================================================================================

namespace kernels {// kernels namespace

template <typename ValueT>
struct RootChild {
    Coord    ijk;// origin of the child node
    uint32_t idx;// linear offset for the child node
    ValueT   val[2];// first and last values as defined by child->getFirstValue/getLastValue
    RootChild(Coord i=Coord(), uint32_t j=0) : ijk(i), idx(j) {};// c-tor
    bool operator()(const RootChild &a, const RootChild &b) const { return a.ijk < b.ijk; }// for sorting
};

// CPU kernel!
template<typename BuildT>
void processRoot(NanoTree<BuildT> *d_tree)
{// the root needs special care since unlike other nodes it's sparse and not dense!
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using TileT  = typename RootT::Tile;
    using ValueT = typename RootT::ValueType;
    using ChildT = RootChild<ValueT>;
    static const int dim = int(RootT::ChildNodeType::DIM);

    // First copy the tree and root and then its tiles, which is of unknown size
    nanovdb::cuda::UnifiedBuffer uBuffer(sizeof(TreeT) + sizeof(RootT), sizeof(TreeT) + sizeof(RootT) + 64*sizeof(TileT));
    cudaCheck(cudaMemcpy(uBuffer.data(), d_tree, uBuffer.size(), cudaMemcpyDeviceToHost));// copy Tree and Root (minus tiles)
    if (!uBuffer.data<TreeT>()->isRootNext()) throw std::runtime_error("ERROR: expected no padding between tree and root!");
    if ( uBuffer.data<TreeT>()->root().tileCount() == 0) return;// empty root node so nothing to do
    uBuffer.resize(sizeof(TreeT) + uBuffer.data<TreeT>()->root().memUsage());// likely does nothing since we reserved 64 tiles
    RootT *root = &uBuffer.data<TreeT>()->root();
    cudaCheck(cudaMemcpy(root + 1, (char*)(d_tree + 1) + sizeof(RootT), root->tileCount()*sizeof(TileT), cudaMemcpyDeviceToHost));// copy tiles

    // Sort the child nodes of the root in lexicographic order
    nanovdb::cuda::UnifiedBuffer nodeBuffer(root->tileCount()*sizeof(ChildT));// potential over-allocation
    auto *first = nodeBuffer.data<ChildT>(), *last = first;
    for (auto it=root->beginChild(); it; ++it) *last++ = ChildT(it.getCoord(), it.pos());
    if (last - first < 2) return;// zero or one child node so nothing to do!
    std::sort(first, last, ChildT());// lexicographic ordering

    // We employ a simple z-scanline algorithm that inserts inactive tiles with
    // the inside value if they are sandwiched between inside child nodes only!
    for (ChildT *a = first, *b = a+1; b!=last; ++a, ++b) {// loop over pairs of adjacent child nodes
        const Coord d = b->ijk - a->ijk;// coord delta of adjacent child nodes
        if (d[0]!=0 || d[1]!=0 || d[2]==dim) continue;// not same z-scanline or they are neighbors
        util::cuda::lambdaKernel<<<1, 1>>>(1, [=] __device__(size_t) {
            a->val[1] = root->getChild(root->tile(a->idx))->getLastValue();
            b->val[0] = root->getChild(root->tile(b->idx))->getFirstValue();
        });
        cudaCheck(cudaDeviceSynchronize());// required for host access to RootChild::val[2]
        if (a->val[1] > 0 || b->val[0] > 0) continue; // scanline is not inside a surface
        for (Coord c = a->ijk.offsetBy(0,0,dim); c[2] != b->ijk[2]; c[2] += dim) {
            TileT *tile = root->probeTile(c);
            if (!tile) throw std::runtime_error("ERROR: missing internal tile! Please add them and try again!");
            tile->setValue(c, false, -root->background());
        }
    }
    cudaCheck(cudaMemcpy(d_tree + 1, root, root->memUsage(), cudaMemcpyHostToDevice));// copy tiles back to device
}// processRoot

//================================================================================================

template<typename BuildT, int LEVEL>
__global__ void processNode(NanoTree<BuildT> *d_tree, size_t count)
{
    using NodeT = typename NanoNode<BuildT, LEVEL>::type;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t nValue = tid & (NodeT::SIZE - 1u);
    auto &node = *(d_tree->template getFirstNode<LEVEL>() + (tid >> (3*NodeT::LOG2DIM)));
    const auto &mask = node.childMask();
    if (mask.isOn(nValue)) return;// ignore if child
    auto value = d_tree->background();// initiate to outside value
    auto n = mask.template findNext<true>(nValue);
    if (n < NodeT::SIZE) {
        if (node.getChild(n)->getFirstValue() < 0) value = -value;
    } else if ((n = mask.template findPrev<true>(nValue)) < NodeT::SIZE) {
        if (node.getChild(n)->getLastValue()  < 0) value = -value;
    } else if (node.getValue(0)<0) {
        value = -value;
    }
    node.setValue(nValue, value);
}// processNode

//================================================================================================

template<typename BuildT>
__global__ void processLeaf(NanoTree<BuildT> *d_tree, size_t count)
{
    using LeafT = NanoLeaf<BuildT>;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t nVoxel = tid & (LeafT::SIZE - 1u);
    auto *leaf = d_tree->getFirstLeaf() + (tid >> (3*LeafT::LOG2DIM));
    const auto &mask = leaf->valueMask();
    if (mask.isOn(nVoxel)) return;
    auto *buffer = leaf->mValues;
    auto n = mask.template findNext<true>(nVoxel);
    if (n == LeafT::SIZE && (n = mask.template findPrev<true>(nVoxel)) == LeafT::SIZE) n = 0u;
    buffer[nVoxel] = buffer[n]<0 ? -d_tree->background() : d_tree->background();
}// processLeaf

//================================================================================================

template <typename BuildT>
__global__ void cpyNodeCountKernel(NanoGrid<BuildT> *d_grid, uint64_t *d_count)
{
    NANOVDB_ASSERT(d_grid->isSequential());
    for (int i=0; i<3; ++i) *d_count++ = d_grid->tree().nodeCount(i);
    *d_count = d_grid->tree().root().tileCount();
}

}// kernels namespace

//================================================================================================

template <typename BuildT>
void SignedFloodFill<BuildT>::operator()(NanoGrid<BuildT> *d_grid)
{
    static_assert(BuildTraits<BuildT>::is_float, "cuda::SignedFloodFill only works on float grids");
    NANOVDB_ASSERT(d_grid);
    uint64_t count[4], *d_count = nullptr;
    cudaCheck(util::cuda::mallocAsync((void**)&d_count, 4*sizeof(uint64_t), mStream));
    kernels::cpyNodeCountKernel<BuildT><<<1, 1, 0, mStream>>>(d_grid, d_count);
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&count, d_count, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(util::cuda::freeAsync(d_count, mStream));

    static const int threadsPerBlock = 128;
    auto blocksPerGrid = [&](size_t count)->uint32_t{return (count + (threadsPerBlock - 1)) / threadsPerBlock;};
    auto *d_tree = reinterpret_cast<NanoTree<BuildT>*>(d_grid + 1);

    if (mVerbose) mTimer.start("\nProcess leaf nodes");
    kernels::processLeaf<BuildT><<<blocksPerGrid(count[0]<<9), threadsPerBlock, 0, mStream>>>(d_tree, count[0]<<9);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process lower internal nodes");
    kernels::processNode<BuildT,1><<<blocksPerGrid(count[1]<<12), threadsPerBlock, 0, mStream>>>(d_tree, count[1]<<12);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process upper internal nodes");
    kernels::processNode<BuildT,2><<<blocksPerGrid(count[2]<<15), threadsPerBlock, 0, mStream>>>(d_tree, count[2]<<15);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process root node");
    kernels::processRoot<BuildT>(d_tree);
    if (mVerbose) mTimer.stop();
    cudaCheckError();
}// SignedFloodFill::operator()

//================================================================================================

template<typename BuildT>
typename util::enable_if<BuildTraits<BuildT>::is_float, void>::type
signedFloodFill(NanoGrid<BuildT> *d_grid, bool verbose, cudaStream_t stream)
{
    SignedFloodFill<BuildT> sff(verbose, stream);
    sff(d_grid);
    auto *d_gridData = d_grid->data();
    Checksum cs = getChecksum(d_gridData, stream);
    if (cs.isFull()) {// CheckMode::Partial checksum is unaffected
        updateChecksum(d_gridData, CheckMode::Full, stream);
    }
    cudaCheck(cudaStreamSynchronize(stream));
}

}// namespace tools::cuda

template<typename BuildT>
[[deprecated("Use nanovdb::tools::cuda::signedFloodFill instead.")]]
typename util::enable_if<BuildTraits<BuildT>::is_float, void>::type
cudaSignedFloodFill(NanoGrid<BuildT> *d_grid, bool verbose = false, cudaStream_t stream = 0)
{
    return tools::cuda::signedFloodFill<BuildT>(d_grid, verbose, stream);
}

}// namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_SIGNEDFLOODFILL_CUH_HAS_BEEN_INCLUDED
