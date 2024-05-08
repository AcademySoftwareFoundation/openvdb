#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/convolution/pack_info/PackInfoOps.h"
#include "detail/utils/nanovdb/CustomAccessors.h"


namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType>
__hostdev__ inline void convolutionKernelMapVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                          BatchGridAccessor<GridType> targetBatchAcc,
                                                          BatchGridAccessor<GridType> sourceBatchAcc,
                                                          TorchRAcc32<int32_t, 2> kmap,
                                                          const nanovdb::Coord& kernelStart, const nanovdb::Coord& kernelSize,
                                                          const nanovdb::Coord& stride) {
    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    const nanovdb::NanoGrid<GridType>* source = sourceBatchAcc.grid(batchIdx);
    const nanovdb::NanoGrid<GridType>* target = targetBatchAcc.grid(batchIdx);

    const LeafNodeType& leaf = target->tree().template getFirstNode<0>()[leafIdx];
    auto sourceAcc = source->getAccessor();

    const int64_t targetBaseOffset = targetBatchAcc.voxelOffset(batchIdx);
    const int64_t sourceBaseOffset = sourceBatchAcc.voxelOffset(batchIdx);

    const nanovdb::Coord tCoord = leaf.offsetToGlobalCoord(voxelIdx);
    const nanovdb::Coord sCoord(tCoord.x() * stride.z(), tCoord.y() * stride.y(), tCoord.z() * stride.x());

    const int kIdx = channelIdx; // = kx + ky * ks[0] + kz * ks[1] * ks[0];
    auto ks0ks1 = kernelSize.x() * kernelSize.y();
    const int kz = kernelStart.z() + (channelIdx / ks0ks1);
    const int ky = kernelStart.y() + ((channelIdx % ks0ks1) / kernelSize.x());
    const int kx = kernelStart.x() + ((channelIdx % ks0ks1) % kernelSize.x());

    if (leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx)) {
        const nanovdb::Coord sOffset = sCoord + nanovdb::Coord(kz, ky, kx);
        if (sourceAcc.template get<ActiveOrUnmasked<GridType>>(sOffset)) {
            kmap[targetBaseOffset + leaf.getValue(voxelIdx) - 1][kIdx] = sourceBaseOffset + sourceAcc.getValue(sOffset) - 1;
        } else {
            kmap[targetBaseOffset + leaf.getValue(voxelIdx) - 1][kIdx] = -1;
        }
    }
}


template <typename GridType>
void convolutionKernelMapCPU(const GridBatchImpl::Accessor<GridType>& sourceGridBatchAcc,
                             const GridBatchImpl::Accessor<GridType>& targetGridBatchAcc,
                             const nanovdb::Coord& kernelSize,
                             const nanovdb::Coord& stride,
                             torch::TensorAccessor<int, 2> outKernelMap) {

    const nanovdb::Coord kernelStart ({(int) std::floor(-kernelSize.x() / 2.0 + 1),
                                       (int) std::floor(-kernelSize.y() / 2.0 + 1),
                                       (int) std::floor(-kernelSize.z() / 2.0 + 1) });

    for (size_t bi = 0; bi < sourceGridBatchAcc.batchSize(); bi += 1) {
        const auto* sourceGrid = sourceGridBatchAcc.grid(bi);
        const auto* targetGrid = targetGridBatchAcc.grid(bi);

        const int64_t targetBaseOffset = targetGridBatchAcc.voxelOffset(bi);
        const int64_t sourceBaseOffset = sourceGridBatchAcc.voxelOffset(bi);

        auto sourceAcc = sourceGrid->getAccessor();

        for (auto it = ActiveVoxelIterator<GridType, -1>(targetGrid->tree(), false, targetBaseOffset); it.isValid(); it++) {
            // Note that stride and kernelSize is in DHW
            const nanovdb::Coord sCoord(it->first.x() * stride.z(), it->first.y() * stride.y(), it->first.z() * stride.x());
            // Center kernel is in the middle -- allows for acceleration in Conv.
            int kIdx = 0;
            for (int kz = kernelStart.z(); kz < kernelStart.z() + kernelSize.z(); ++kz) {
                for (int ky = kernelStart.y(); ky < kernelStart.y() + kernelSize.y(); ++ky) {
                    for (int kx = kernelStart.x(); kx < kernelStart.x() + kernelSize.x(); ++kx, ++kIdx) {
                        const nanovdb::Coord& sOffset = sCoord + nanovdb::Coord(kz, ky, kx);
                        if (sourceAcc.template get<ActiveOrUnmasked<GridType>>(sOffset)) {
                            outKernelMap[it->second][kIdx] = sourceAcc.getValue(sOffset) - 1 + sourceBaseOffset;
                        } else {
                            outKernelMap[it->second][kIdx] = -1;
                        }
                    }
                }
            }
        }
    }
}


template <>
void dispatchConvolutionKernelMap<torch::kCUDA>(const GridBatchImpl& sourceBatchHdl,
                                                const GridBatchImpl& targetBatchHdl,
                                                torch::Tensor& kernelMap,
                                                const Vec3iOrScalar& kernelSize,
                                                const Vec3iOrScalar& stride) {

    const nanovdb::Coord& kernelSizeCoord = kernelSize.value();
    const nanovdb::Coord& strideCoord = stride.value();
    FVDB_DISPATCH_GRID_TYPES(sourceBatchHdl, [&]() {
        const nanovdb::Coord kernelStart ({(int) std::floor(-kernelSizeCoord.x() / 2.0 + 1),
                                           (int) std::floor(-kernelSizeCoord.y() / 2.0 + 1),
                                           (int) std::floor(-kernelSizeCoord.z() / 2.0 + 1) });

        auto sourceBatchAccessor = sourceBatchHdl.deviceAccessor<GridType>();
        auto targetBatchAccessor = targetBatchHdl.deviceAccessor<GridType>();
        auto kernelMapAcc = kernelMap.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, BatchGridAccessor<GridType> batchAcc) {
            convolutionKernelMapVoxelCallback<GridType>(bidx, lidx, vidx, cidx, batchAcc, sourceBatchAccessor, kernelMapAcc, kernelStart, kernelSizeCoord, strideCoord);
        };
        forEachVoxelCUDA<GridType>(128, kernelSizeCoord.x() * kernelSizeCoord.y() * kernelSizeCoord.z(), targetBatchHdl, cb);
    });
}

template <>
void dispatchConvolutionKernelMap<torch::kCPU>(const GridBatchImpl& source,
                                               const GridBatchImpl& target,
                                               torch::Tensor& kernelMap,
                                               const Vec3iOrScalar& kernelSize,
                                               const Vec3iOrScalar& stride) {

    const nanovdb::Coord& kernelSizeCoord = kernelSize.value();
    const nanovdb::Coord& strideCoord = stride.value();
    FVDB_DISPATCH_GRID_TYPES(source, [&]() {
        convolutionKernelMapCPU<GridType>(source.hostAccessor<GridType>(),
                                          target.hostAccessor<GridType>(),
                                          kernelSizeCoord, strideCoord,
                                          kernelMap.accessor<int, 2>());
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
