#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/nanovdb/CustomAccessors.h"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor>
__hostdev__ inline void setMaskedIjkCallback(int32_t bidx, int32_t eidx,
                                             JaggedAccessor<ScalarType, 2> coords,
                                             BatchGridAccessor<nanovdb::ValueOnIndexMask> batchAccessor,
                                             bool maskedState) {
    const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* gpuGrid = batchAccessor.grid(bidx);
    auto acc = gpuGrid->getAccessor();

    const auto coord = coords.data()[eidx];
    const nanovdb::Coord ijk(coord[0], coord[1], coord[2]);

    if constexpr (DeviceTag == torch::kCUDA) {
        acc.template set<typename fvdb::AtomicMaskedStateSetOnlyDevice>(ijk, maskedState); // false means we disable voxels
    } else {
        acc.template set<typename fvdb::AtomicMaskedStateSetOnlyHost>(ijk, maskedState); // false means we disable voxels
    }
}


template <c10::DeviceType DeviceTag>
void SetMaskedIjk(const GridBatchImpl& batchHdl,
                  const JaggedTensor& ijk,
                  bool maskedState) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(ijk);
    TORCH_CHECK(batchHdl.isMutable(), "Cannot disable voxels in an immutable grid");
    TORCH_CHECK_VALUE(ijk.dim() == 2,
                      "Expected 2 dimensional ijk with shape (N, 3) but got = " +
                      std::to_string(ijk.jdata().dim()) + " dimensional ijk.");
    TORCH_CHECK_VALUE(ijk.jdata().size(1) == 3,
                      "Expected 3 dimensional ijk but got points.shape[1] = " +
                      std::to_string(ijk.jdata().size(1)));
    if(ijk.jdata().size(0) == 0) {
        return; // nothing to do
    }
    TORCH_CHECK(ijk.size(0) > 0, "Empty tensor (ijk)");

    AT_DISPATCH_INTEGRAL_TYPES(ijk.scalar_type(), "SetMaskedIjk", [&]() {
        auto batchAcc = gridBatchAccessor<DeviceTag, nanovdb::ValueOnIndexMask>(batchHdl);

        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> cA) {
                setMaskedIjkCallback<scalar_t, DeviceTag, JaggedRAcc32>(bidx, eidx, cA, batchAcc, maskedState);
            };
            forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, ijk, cb);
        } else {
            auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> cA) {
                setMaskedIjkCallback<scalar_t, DeviceTag, JaggedAcc>(bidx, eidx, cA, batchAcc, maskedState);
            };
            forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
        }
    });
}



template <>
void dispatchSetMaskedIjk<torch::kCUDA>(const GridBatchImpl& batchHdl, const JaggedTensor& coords, bool maskedState) {
    SetMaskedIjk<torch::kCUDA>(batchHdl, coords, maskedState);
}

template <>
void dispatchSetMaskedIjk<torch::kCPU>(const GridBatchImpl& batchHdl, const JaggedTensor& coords, bool maskedState) {
    SetMaskedIjk<torch::kCPU>(batchHdl, coords, maskedState);
}

} // namespace ops
} // namespace detail
} // namespace fvdb