// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/cuda/SignedFloodFill.cuh

    \author Ken Museth

    \date May 3, 2023

    \brief Performs signed flood-fill operation on the hierarchical tree structure on the device

    \todo This tools needs to handle the (extremely) rare case when root node
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
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>

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

namespace {// anonymous namespace

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

template<typename BuildT>
__global__ void processRootKernel(NanoTree<BuildT> *tree)
{
    // auto &root = tree->root();
    /*
    using ChildT = typename RootT::ChildNodeType;
    // Insert the child nodes into a map sorted according to their origin
    std::map<Coord, ChildT*> nodeKeys;
    typename RootT::ChildOnIter it = root.beginChildOn();
    for (; it; ++it) nodeKeys.insert(std::pair<Coord, ChildT*>(it.getCoord(), &(*it)));
    static const Index DIM = RootT::ChildNodeType::DIM;

    // We employ a simple z-scanline algorithm that inserts inactive tiles with
    // the inside value if they are sandwiched between inside child nodes only!
    typename std::map<Coord, ChildT*>::const_iterator b = nodeKeys.begin(), e = nodeKeys.end();
    if ( b == e ) return;
    for (typename std::map<Coord, ChildT*>::const_iterator a = b++; b != e; ++a, ++b) {
        Coord d = b->first - a->first; // delta of neighboring coordinates
        if (d[0]!=0 || d[1]!=0 || d[2]==Int32(DIM)) continue;// not same z-scanline or neighbors
        const ValueT fill[] = { a->second->getLastValue(), b->second->getFirstValue() };
        if (!(fill[0] < 0) || !(fill[1] < 0)) continue; // scanline isn't inside
        Coord c = a->first + Coord(0u, 0u, DIM);
        for (; c[2] != b->first[2]; c[2] += DIM) root.addTile(c, mInside, false);
    }
    */
    //root.setBackground(mOutside, /*updateChildNodes=*/false);
}// processRootKernel

//================================================================================================

template<typename BuildT, int LEVEL>
__global__ void processNodeKernel(NanoTree<BuildT> *tree, size_t count)
{
    using NodeT = typename NanoNode<BuildT, LEVEL>::type;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t nValue = tid & (NodeT::SIZE - 1u);
    auto &node = *(tree->template getFirstNode<LEVEL>() + (tid >> (3*NodeT::LOG2DIM)));
    const auto &mask = node.childMask();
    if (mask.isOn(nValue)) return;// ignore if child
    auto value = tree->background();// initiate to outside value
    auto n = mask.template findNext<true>(nValue);
    if (n < NodeT::SIZE) {
        if (node.getChild(n)->getFirstValue() < 0) value = -value;
    } else if ((n = mask.template findPrev<true>(nValue)) < NodeT::SIZE) {
        if (node.getChild(n)->getLastValue()  < 0) value = -value;
    } else if (node.getValue(0)<0) {
        value = -value;
    }
    node.setValue(nValue, value);
}// processNodeKernel

//================================================================================================

template<typename BuildT>
__global__ void processLeafKernel(NanoTree<BuildT> *tree, size_t count)
{
    using LeafT = NanoLeaf<BuildT>;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t nVoxel = tid & (LeafT::SIZE - 1u);
    auto *leaf = tree->getFirstLeaf() + (tid >> (3*LeafT::LOG2DIM));
    const auto &mask = leaf->valueMask();
    if (mask.isOn(nVoxel)) return;
    auto *buffer = leaf->mValues;
    auto n = mask.template findNext<true>(nVoxel);
    if (n == LeafT::SIZE && (n = mask.template findPrev<true>(nVoxel)) == LeafT::SIZE) n = 0u;
    buffer[nVoxel] = buffer[n]<0 ? -tree->background() : tree->background();
}// processLeafKernel

//================================================================================================

template <typename BuildT>
__global__ void cpyNodeCountKernel(NanoGrid<BuildT> *d_grid, uint64_t *d_count)
{
    NANOVDB_ASSERT(d_grid->isSequential());
    for (int i=0; i<3; ++i) *d_count++ = d_grid->tree().nodeCount(i);
    *d_count = d_grid->tree().root().tileCount();
}

}// anonymous namespace

//================================================================================================

template <typename BuildT>
void SignedFloodFill<BuildT>::operator()(NanoGrid<BuildT> *d_grid)
{
    static_assert(BuildTraits<BuildT>::is_float, "cuda::SignedFloodFill only works on float grids");
    NANOVDB_ASSERT(d_grid);
    uint64_t count[4], *d_count = nullptr;
    cudaCheck(util::cuda::mallocAsync((void**)&d_count, 4*sizeof(uint64_t), mStream));
    cpyNodeCountKernel<BuildT><<<1, 1, 0, mStream>>>(d_grid, d_count);
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&count, d_count, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(util::cuda::freeAsync(d_count, mStream));

    static const int threadsPerBlock = 128;
    auto blocksPerGrid = [&](size_t count)->uint32_t{return (count + (threadsPerBlock - 1)) / threadsPerBlock;};
    auto *tree = reinterpret_cast<NanoTree<BuildT>*>(d_grid + 1);

    if (mVerbose) mTimer.start("\nProcess leaf nodes");
    processLeafKernel<BuildT><<<blocksPerGrid(count[0]<<9), threadsPerBlock, 0, mStream>>>(tree, count[0]<<9);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process lower internal nodes");
    processNodeKernel<BuildT,1><<<blocksPerGrid(count[1]<<12), threadsPerBlock, 0, mStream>>>(tree, count[1]<<12);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process upper internal nodes");
    processNodeKernel<BuildT,2><<<blocksPerGrid(count[2]<<15), threadsPerBlock, 0, mStream>>>(tree, count[2]<<15);
    cudaCheckError();

    //if (mVerbose) mTimer.restart("Process root node");
    //processRootKernel<BuildT><<<1, 1, 0, mStream>>>(tree);
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
