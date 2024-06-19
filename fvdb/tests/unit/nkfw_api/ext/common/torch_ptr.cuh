#ifndef TORCH_PTR_CUH
#define TORCH_PTR_CUH

#include <torch/torch.h>

#define float1in packed_accessor32<float, 1, torch::RestrictPtrTraits>
#define float2in packed_accessor32<float, 2, torch::RestrictPtrTraits>
#define float3in packed_accessor32<float, 3, torch::RestrictPtrTraits>
using Float1Accessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;
using Float2Accessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using Float3Accessor = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;

#define int1in packed_accessor32<int, 1, torch::RestrictPtrTraits>
#define int2in packed_accessor32<int, 2, torch::RestrictPtrTraits>
#define int3in packed_accessor32<int, 3, torch::RestrictPtrTraits>
using Int1Accessor = torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>;
using Int2Accessor = torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>;
using Int3Accessor = torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>;

#define long1in packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>
#define long2in packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>
#define long3in packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>
using Long1Accessor = torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>;
using Long2Accessor = torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits>;
using Long3Accessor = torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>;

#endif //TORCH_PTR_CUH
