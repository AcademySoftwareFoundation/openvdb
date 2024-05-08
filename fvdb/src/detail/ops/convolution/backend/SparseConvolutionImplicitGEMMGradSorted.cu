#include <torch/extension.h>
#include <cuda_fp16.h>
#include "detail/ops/Ops.h"
#include "detail/ops/convolution/backend/ConvOps.h"


namespace fvdb {
namespace detail {
namespace ops {

template <int bytes>
struct global_load;

template <>
struct global_load<16>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint4 &data = *reinterpret_cast<uint4 *>(&D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)(pred_guard & 1)), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <>
struct global_load<8>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint2 const *ptr_ldg = reinterpret_cast<uint2 const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 2; ldg_idx++)
    {
      uint2 &data = *(reinterpret_cast<uint2 *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %3, 0;\n"
          "  mov.b32 %0, %4;\n"
          "  mov.b32 %1, %5;\n"
          "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
          "}\n"
          : "=r"(data.x), "=r"(data.y)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data.x), "r"(data.y));
    }
  }
};

template <>
struct global_load<4>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    unsigned const *ptr_ldg = reinterpret_cast<unsigned const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 4; ldg_idx++)
    {
      unsigned &data = *(reinterpret_cast<unsigned *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %2, 0;\n"
          "  mov.b32 %0, %3;\n"
          "  @p ld.global.u32 %0, [%1];\n"
          "}\n"
          : "=r"(data)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data));
    }
  }
};

template <>
struct global_load<2>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint16_t const *ptr_ldg = reinterpret_cast<uint16_t const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 8; ldg_idx++)
    {
      uint16_t &data = *(reinterpret_cast<uint16_t *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %2, 0;\n"
          "  mov.b16 %0, %3;\n"
          "  @p ld.global.u16 %0, [%1];\n"
          "}\n"
          : "=h"(data)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "h"(data));
    }
  }
};

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}


// conv_backward_cuda_m16n16k64_m16n16k64_m16n16k16_f16f16f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, half *__restrict__ A, half *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, half *__restrict__ C)
{
  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_warp[8];
  __shared__ half A_shared[2560];
  __shared__ half B_shared[2560];
  half A_shared_warp[8];
  half B_shared_warp[8];
  for (int i = 0; i < 8; ++i)
  {
    C_warp[0 + i] = 0.0;
  }

  // hoisting shared pointer offsets
  int j_factors1 = (N + 15) / 16 / 1;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original + 15) / 16 * kernel_volume * j_factors1);
  half *cur_C = C + blockIdx_z * kernel_volume * K_original * N;
  int* out_in_map_ptr = out_in_map
      + (threadIdx.y * 16
      + threadIdx.x / 2
    ) * kernel_volume
    + ((threadIdx.y * 256) % 16) / K_tile_padded
    + ((threadIdx.x * 8) % 16) / K_tile_padded
    + (blockIdx_y / j_factors1 * 16) / K_tile_padded;
  half* A_ptr = A
    + ((threadIdx.y * 256 % 16) % K_tile_padded)
    + ((threadIdx.x * 8 % 16) % K_tile_padded)
    + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
  half* B_ptr = B
    + (blockIdx_y % j_factors1) * 16
    + (threadIdx.x * 8) % 16;
  int reorder_offset = threadIdx.y * 256 / 16
    + threadIdx.x * 8 / 16;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / ((K_original + K_tile - 1) / K_tile);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;
  int cur_C_ic_start = (blockIdx_y / j_factors1 * 16) % K_tile_padded + (threadIdx.x / 4);
  int cur_C_oc_start = (blockIdx_y % j_factors1) * 16 + threadIdx.y / 1 * 16 + (threadIdx.x % 4) * 2;
  half *C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check)
  {
    int A_ld_start = ((threadIdx.y * 256 % 16) % K_tile_padded) + ((threadIdx.x * 8 % 16) % K_tile_padded) + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
    int A_ld_amount = min(A_ld_start + 8, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 2);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;
  if constexpr (N_ld_check)
  {
    int B_ld_start = (blockIdx_y % j_factors1) * 16 + (threadIdx.x * 8) % 16;
    int B_ld_amount = min(B_ld_start + 8, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 2);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;


  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = A_loaded;
      }
      else
      {
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      uint4 B_loaded = make_uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) =
          B_loaded;
    }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif __CUDA_ARCH__ >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }

  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = A_loaded;
      }
      else
      {
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) =
          B_loaded;
        }
        else
        {
          *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(0, 0, 0, 0);
        }
      }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif __CUDA_ARCH__ >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }

  for (int local_id = 0; local_id < 8; ++local_id)
  {
    if constexpr (K_ld_check || N_ld_check)
    {
      if (cur_C_ic_start + ((local_id / 2) % 2) * 8 < K_original && cur_C_oc_start + (local_id % 2) + (local_id / 4) * 8 < N)
        C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[0 + local_id]);
    }
    else
    {
      C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[0 + local_id]);
    }
  }
}


// conv_backward_cuda_m32n64k64_m32n32k64_m16n16k16_f16f16f32
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, half *__restrict__ A, half *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, half *__restrict__ C)
{
  float C_warp[32];
  __shared__ half A_shared[2560];
  __shared__ half B_shared[4608];
  half A_shared_warp[16];
  half B_shared_warp[16];
  for (int i0_0_3_init = 0; i0_0_3_init < 2; ++i0_0_3_init)
  {
    for (int i1_0_4_init = 0; i1_0_4_init < 2; ++i1_0_4_init)
    {
      for (int i = 0; i < 8; ++i)
      {
        C_warp[((i0_0_3_init * 16) + (i1_0_4_init * 8)) + i] = 0.0;
      };
    }
  }

  // hoisting shared pointer offsets
  int j_factors1 = N / 16 / 4;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original * kernel_volume + 31) / 32 * j_factors1);
  half *cur_C = C + blockIdx_z * kernel_volume * N * K_original;
  int* out_in_map_ptr = out_in_map
      + (threadIdx.y * 8
      + threadIdx.x / 4
    ) * kernel_volume
    + ((threadIdx.y * 256) % 32) / K_original
    + ((threadIdx.x * 8) % 32) / K_original
    + (blockIdx_y / j_factors1 * 32) / K_original;
  half* A_ptr = A
    + ((threadIdx.y * 256 % 32) % K_original)
    + ((threadIdx.x * 8 % 32) % K_original)
    + ((blockIdx_y / j_factors1 * 32) % K_original);
  half* B_ptr = B
    + (blockIdx_y % j_factors1) * 64
    + (threadIdx.x * 8) % 64;
  int reorder_offset = threadIdx.y * 256 / 64
    + threadIdx.x * 8 / 64;
  half* C_ptr = cur_C
    + blockIdx_x / 1 * 108 * N / 16 * 256
    + blockIdx_y / j_factors1 * 2 * N / 16 * 256
    + (threadIdx.y % 1) * 2 * N / 16 * 256
    + (blockIdx_x % 1) * j_factors1 * 64
    + (blockIdx_y % j_factors1) * 64
    + threadIdx.y / 1 * 32
    + (threadIdx.x % 4) * 2
    + (threadIdx.x / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(uint4*)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(uint4 *)(B_ptr + v0 * N);
    }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(((i2_0_1 * 1152) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 72) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif __CUDA_ARCH__ >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }

  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(uint4*)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(uint4*)(B_ptr + v0 * N);
        }
        else
        {
          *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = make_uint4(0, 0, 0, 0);
        }
      }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(((i2_0_1 * 1152) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 72) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif __CUDA_ARCH__ >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }

  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
  {

    half* C_ptr_local = C_ptr + ax0_0 * N / 16 * 256;

    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        C_ptr_local[ax1_0_2 * 16 + (((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[((ax0_0 * 16) + (ax1_0_2 * 8)) + local_id]);
      };
    }
  }
}


// conv_backward_cuda_m16n16k64_m16n16k64_m16n16k16_tf32tf32f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_tf32tf32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, float *__restrict__ C)
{
  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_warp[8];
  __shared__ float A_shared[2560];
  __shared__ float B_shared[2560];
  float A_shared_warp[8];
  float B_shared_warp[8];
  for (int i = 0; i < 8; ++i)
  {
    C_warp[0 + i] = 0.0;
  }

  // hoisting shared pointer offsets
  int j_factors1 = (N + 15) / 16 / 1;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original + 15) / 16 * kernel_volume * j_factors1);
  float *cur_C = C + blockIdx_z * kernel_volume * K_original * N;
  int* out_in_map_ptr = out_in_map
      + (threadIdx.y * 16
      + threadIdx.x / 2
    ) * kernel_volume
    + ((threadIdx.y * 256) % 16) / K_tile_padded
    + ((threadIdx.x * 8) % 16) / K_tile_padded
    + (blockIdx_y / j_factors1 * 16) / K_tile_padded;
  float* A_ptr = A
    + ((threadIdx.y * 256 % 16) % K_tile_padded)
    + ((threadIdx.x * 8 % 16) % K_tile_padded)
    + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
  float* B_ptr = B
    + (blockIdx_y % j_factors1) * 16
    + (threadIdx.x * 8) % 16;
  int reorder_offset = threadIdx.y * 256 / 16
    + threadIdx.x * 8 / 16;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / ((K_original + K_tile - 1) / K_tile);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;
  int cur_C_ic_start = (blockIdx_y / j_factors1 * 16) % K_tile_padded + (threadIdx.x / 4);
  int cur_C_oc_start = (blockIdx_y % j_factors1) * 16 + threadIdx.y / 1 * 16 + (threadIdx.x % 4) * 2;
  float *C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check)
  {
    int A_ld_start = ((threadIdx.y * 256 % 16) % K_tile_padded) + ((threadIdx.x * 8 % 16) % K_tile_padded) + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
    int A_ld_amount = min(A_ld_start + 8, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 4);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    // load twice
    A_pred_guard = 3;
  if constexpr (N_ld_check)
  {
    int B_ld_start = (blockIdx_y % j_factors1) * 16 + (threadIdx.x * 8) % 16;
    int B_ld_amount = min(B_ld_start + 8, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 4);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    // load twice
    B_pred_guard = 3;


  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        uint4 A_loaded[2] = {make_uint4(0, 0, 0, 0)};
        global_load<K_ld_factor>(A_loaded[0], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        global_load<K_ld_factor>(A_loaded[1], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded) + 4, A_pred_guard >> (4 * 4 / K_ld_factor));
        *(ulonglong4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *reinterpret_cast<ulonglong4 *>(A_loaded);
      }
      else
      {
        *(ulonglong4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_ulonglong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      uint4 B_loaded[2] = {make_uint4(0, 0, 0, 0)};
      global_load<N_ld_factor>(B_loaded[0], B_ptr + v0 * N, B_pred_guard);
      global_load<N_ld_factor>(B_loaded[1], B_ptr + v0 * N + 4, B_pred_guard >> (4 * 4 / N_ld_factor));
      *(ulonglong4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *reinterpret_cast<ulonglong4 *>(B_loaded);
    }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int local_size = 0; local_size < 8; ++local_size)
      {
        A_shared_warp[local_size] = A_shared[(((((i2_0_1 * 640) + ((local_size >> 1) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + ((local_size & 1) * 8)) + (((int)threadIdx.x) >> 2))];
      }
      for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
      {
        B_shared_warp[local_size_1] = B_shared[(((((i2_0_1 * 640) + ((local_size_1 & 3) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + ((local_size_1 >> 2) * 8)) + (((int)threadIdx.x) >> 2))];
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(A_shared_warp + 4))[2]), "r"(((unsigned *)(A_shared_warp + 4))[3]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "r"(((unsigned *)(B_shared_warp + 2))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(A_shared_warp + 4))[2]), "r"(((unsigned *)(A_shared_warp + 4))[3]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "r"(((unsigned *)(B_shared_warp + 6))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
    }
  }

  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded
      ];

      if (input_idx != -1)
      {
        uint4 A_loaded[2] = {make_uint4(0, 0, 0, 0)};
        global_load<K_ld_factor>(A_loaded[0], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        global_load<K_ld_factor>(A_loaded[1], A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded) + 4, A_pred_guard >> (4 * 4 / K_ld_factor));
        *(ulonglong4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *reinterpret_cast<ulonglong4 *>(A_loaded);
      }
      else
      {
        *(ulonglong4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_ulonglong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        uint4 B_loaded[2] = {make_uint4(0, 0, 0, 0)};
        global_load<N_ld_factor>(B_loaded[0], B_ptr + v0 * N, B_pred_guard);
        global_load<N_ld_factor>(B_loaded[1], B_ptr + v0 * N + 4, B_pred_guard >> (4 * 4 / N_ld_factor));
        *(ulonglong4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = *reinterpret_cast<ulonglong4 *>(B_loaded);
      }
      else
      {
        *(ulonglong4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_ulonglong4(0, 0, 0, 0);
      }
    }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int local_size = 0; local_size < 8; ++local_size)
      {
        A_shared_warp[local_size] = A_shared[(((((i2_0_1 * 640) + ((local_size >> 1) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + ((local_size & 1) * 8)) + (((int)threadIdx.x) >> 2))];
      }
      for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
      {
        B_shared_warp[local_size_1] = B_shared[(((((i2_0_1 * 640) + ((local_size_1 & 3) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + ((local_size_1 >> 2) * 8)) + (((int)threadIdx.x) >> 2))];
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(A_shared_warp + 4))[2]), "r"(((unsigned *)(A_shared_warp + 4))[3]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "r"(((unsigned *)(B_shared_warp + 2))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(A_shared_warp + 4))[2]), "r"(((unsigned *)(A_shared_warp + 4))[3]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "r"(((unsigned *)(B_shared_warp + 6))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
    }
  }

  for (int local_id = 0; local_id < 8; ++local_id)
  {
    if constexpr (K_ld_check || N_ld_check)
    {
      if (cur_C_ic_start + ((local_id / 2) % 2) * 8 < K_original && cur_C_oc_start + (local_id % 2) + (local_id / 4) * 8 < N)
        C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[0 + local_id];
    }
    else
    {
      C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[0 + local_id];
    }
  }
}


// conv_backward_cuda_m32n64k64_m32n32k64_m16n16k16_tf32tf32f32
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_tf32tf32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, float *__restrict__ C)
{
  float C_warp[32];
  __shared__ float A_shared[2560];
  __shared__ float B_shared[4608];
  float A_shared_warp[16];
  float B_shared_warp[16];
  for (int i0_0_3_init = 0; i0_0_3_init < 2; ++i0_0_3_init)
  {
    for (int i1_0_4_init = 0; i1_0_4_init < 2; ++i1_0_4_init)
    {
      for (int i = 0; i < 8; ++i)
      {
        C_warp[((i0_0_3_init * 16) + (i1_0_4_init * 8)) + i] = 0.0;
      };
    }
  }

  // hoisting shared pointer offsets
  int j_factors1 = N / 16 / 4;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original * kernel_volume + 31) / 32 * j_factors1);
  float *cur_C = C + blockIdx_z * kernel_volume * N * K_original;
  int* out_in_map_ptr = out_in_map
      + (threadIdx.y * 8
      + threadIdx.x / 4
    ) * kernel_volume
    + ((threadIdx.y * 256) % 32) / K_original
    + ((threadIdx.x * 8) % 32) / K_original
    + (blockIdx_y / j_factors1 * 32) / K_original;
  float* A_ptr = A
    + ((threadIdx.y * 256 % 32) % K_original)
    + ((threadIdx.x * 8 % 32) % K_original)
    + ((blockIdx_y / j_factors1 * 32) % K_original);
  float* B_ptr = B
    + (blockIdx_y % j_factors1) * 64
    + (threadIdx.x * 8) % 64;
  int reorder_offset = threadIdx.y * 256 / 64
    + threadIdx.x * 8 / 64;
  float* C_ptr = cur_C
    + blockIdx_x / 1 * 108 * N / 16 * 256
    + blockIdx_y / j_factors1 * 2 * N / 16 * 256
    + (threadIdx.y % 1) * 2 * N / 16 * 256
    + (blockIdx_x % 1) * j_factors1 * 64
    + (blockIdx_y % j_factors1) * 64
    + threadIdx.y / 1 * 32
    + (threadIdx.x % 4) * 2
    + (threadIdx.x / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_ptr = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_ptr = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(ulonglong4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(ulonglong4*)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(ulonglong4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_ulonglong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      int v0 = reorder_loc_ptr[reorder_offset_inner];
      *(ulonglong4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(ulonglong4 *)(B_ptr + v0 * N);
    }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {
        for (int local_size = 0; local_size < 8; ++local_size)
        {
          A_shared_warp[((ax1_0 * 8) + local_size)] = A_shared[((((((i2_0_1 * 640) + ((local_size >> 1) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + (ax1_0 * 16)) + ((local_size & 1) * 8)) + (((int)threadIdx.x) >> 2))];
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {
        for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
        {
          B_shared_warp[((ax1_0_1 * 8) + local_size_1)] = B_shared[(((((((i2_0_1 * 1152) + ((local_size_1 & 3) * 288)) + ((((int)threadIdx.x) & 3) * 72)) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16)) + ((local_size_1 >> 2) * 8)) + (((int)threadIdx.x) >> 2))];
        }
      }

      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
        }
      }
    }
  }

  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * (K_iters - 1);
    if (i2_0_0 >= (M_fwd + 63) / 64)
      continue;

    int* out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    float* A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;
    bool bit_flag = (bool)(reduced_mask_ptr[i2_0_0] & (1 << bitmask_shift));
    if (!bit_flag) continue;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[
        ax0_ax1_fused_0 * 16 * kernel_volume
        + (ax0_ax1_fused_0 * 512 % 32) / K_original
      ];

      if (input_idx != -1)
      {
        *(ulonglong4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(ulonglong4*)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(ulonglong4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_ulonglong4(0ULL, 0ULL, 0ULL, 0ULL);
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {

      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      if (reorder_offset_inner < M_fwd){
        int v0 = reorder_loc_ptr[reorder_offset_inner];
        *(ulonglong4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(ulonglong4*)(B_ptr + v0 * N);
        }
        else
        {
          *(ulonglong4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = make_ulonglong4(0ULL, 0ULL, 0ULL, 0ULL);
        }
      }

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {
        for (int local_size = 0; local_size < 8; ++local_size)
        {
          A_shared_warp[((ax1_0 * 8) + local_size)] = A_shared[((((((i2_0_1 * 640) + ((local_size >> 1) * 160)) + ((((int)threadIdx.x) & 3) * 40)) + (ax1_0 * 16)) + ((local_size & 1) * 8)) + (((int)threadIdx.x) >> 2))];
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {
        for (int local_size_1 = 0; local_size_1 < 8; ++local_size_1)
        {
          B_shared_warp[((ax1_0_1 * 8) + local_size_1)] = B_shared[(((((((i2_0_1 * 1152) + ((local_size_1 & 3) * 288)) + ((((int)threadIdx.x) & 3) * 72)) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16)) + ((local_size_1 >> 2) * 8)) + (((int)threadIdx.x) >> 2))];
        }
      }

      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[2]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("TF32 kernels will not be compiled.")
#endif
        }
      }
    }
  }

  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
  {

    float* C_ptr_local = C_ptr + ax0_0 * N / 16 * 256;

    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        C_ptr_local[ax1_0_2 * 16 + (((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = C_warp[((ax0_0 * 16) + (ax1_0_2 * 8)) + local_id];
      };
    }
  }
}


// conv_backward_cuda_m16n16k64_f32f32f32_sort
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode1_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, float *__restrict__ C)
{

  int j_factors1 = (N + 15) / 16;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original + 15) / 16 * kernel_volume * j_factors1);

  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_local[8];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];

  #pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    C_local[i] = 0.0;
  }

  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)threadIdx.x;

  int kernel_offset = blockIdx_m / (K_tile_padded / 16);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_local = reorder_loc + split_mask_iter * reorder_loc_len ;
  int* reduced_mask_local = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  int channel_offset = (blockIdx_m * 16 + ((threadIdx_x * 4) % 16)) % K_tile_padded;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (16/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
  int reorder_offset = threadIdx_x /(16/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 4);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 4);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int cur_C_ic_start = (blockIdx_m * 16 + (threadIdx_x / 4)) % K_tile_padded;
  int cur_C_oc_start = blockIdx_n * 16 + (threadIdx_x % 4);
  float * C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check) // IC % cta_M != 0
  {
    int A_ld_start = channel_offset;
    int A_ld_amount = min(A_ld_start + 4, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 4);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;

  if constexpr (N_ld_check) // OC % cta_N != 0
  {
    int B_ld_start = (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
    int B_ld_amount = min(B_ld_start + 4, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 4);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume];
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else
      {
        *(uint4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      int v0 = reorder_loc_local[reorder_offset_inner];
      //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) =
      //    *(float4*)(B_ptr + v0 * N);
      uint4 B_loaded = make_uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume);
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else
      {
        *(uint4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_loc_local[reorder_offset_inner];
        //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) =
        //    *(float4*)(B_ptr + v0 * N);
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;
      }
      else
      {
        *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    int local_row = ((i / 4) * 8);
    int local_col = ((i % 4) * 4);
    if constexpr (K_ld_check || N_ld_check)
    {
      if ( ((cur_C_ic_start + local_row) < K_original) && ((cur_C_oc_start + local_col) < N) )
        C_ptr[local_row * N + local_col] = C_local[i];

    }
    else
    {
      C_ptr[local_row * N + local_col] = C_local[i];
    }
  }
}


// conv_backward_cuda_m32n64k64_f32f32f32_sort
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode1_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, int split_mask_len, int reduced_mask_len, int reorder_loc_len, float *__restrict__ A, float *__restrict__ B, int *__restrict__ reduced_mask, int *__restrict__ out_in_map, int *__restrict__ reorder_loc, float *__restrict__ C)
{

  int j_factors1 = (N + 63) / 64;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original * kernel_volume + 31) / 32 * j_factors1);

  float C_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[4096];

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
    C_local[i] = 0.0;
  }

  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)threadIdx.x;

  int kernel_offset = blockIdx_m / (K_original / 32);
  int split_mask_iter = kernel_offset / split_mask_len;
  int* reorder_loc_local = reorder_loc + split_mask_iter * reorder_loc_len;
  int* reduced_mask_local = reduced_mask + split_mask_iter * reduced_mask_len;
  int bitmask_shift = kernel_offset - split_mask_iter * split_mask_len;

  int channel_offset = (blockIdx_m * 32 + ((threadIdx_x * 4) % 32)) % K_original;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (32/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 64) + ((threadIdx_x * 4) % 64);
  int reorder_offset = threadIdx_x /(64/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 16);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 16);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int C_m_offset = blockIdx_m * 32 + (threadIdx_x / 16);  // C_m_offset
  int C_n_offset = blockIdx_n * 64  + (threadIdx_x % 16);
  // float * C_ptr = cur_C + C_m_offset * N + C_n_offset;

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume];
      if (input_idx != -1)
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =  // ax0_ax1_fused_0 * elements loaded in each loop
            *(float4*)(A_ptr + (input_idx * K_original));
      }
      else
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      int v0 = reorder_loc_local[reorder_offset_inner];
      *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
          *(float4*)(B_ptr + v0 * N);
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    bool bit_flag = (bool)(reduced_mask_local[k_0] & (1 << bitmask_shift));
    if (!bit_flag)
      continue;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume);
      if (input_idx != -1)
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =  // ax0_ax1_fused_0 * elements loaded in each loop
            *(float4*)(A_ptr + (input_idx * K_original));
      }
      else
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_loc_local[reorder_offset_inner];
        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
            *(float4*)(B_ptr + v0 * N);
      }
      else
      {
        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
      int C_m_offset_cur = C_m_offset + ((i / 4) * 4);
      int C_n_offset_cur = C_n_offset + ((i % 4) * 16);
      cur_C[C_m_offset_cur * N + C_n_offset_cur] = C_local[i];
  }
}

template <>
torch::Tensor dispatchSparseConvolutionImplicitGEMMGradSorted<torch::kCUDA>(
    torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _out_in_map, torch::Tensor _reduced_mask,
    torch::Tensor _reorder_loc, const int split_k_iters,
    bool allow_tf32, bool allow_fp16)
{
  bool is_tf = allow_tf32;
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  int kernel_volume = _out_in_map.size(1);
  int split_mask_num = _reorder_loc.size(0);
  int split_mask_len = (kernel_volume + split_mask_num - 1) / split_mask_num;
  int reduced_mask_len = _reduced_mask.size(1);
  int reorder_loc_len = _reorder_loc.size(1);
  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({split_k_iters, num_in_channels * kernel_volume, _kernel.size(1)}, options);
  int num_out_feats = _out_feats.size(1);
  int num_out_channels = _out_feats.size(2);
  auto reduced_mask = _reduced_mask.data_ptr<int>();
  auto out_in_map = _out_in_map.data_ptr<int>();
  auto reorder_loc = _reorder_loc.data_ptr<int>();
  bool is_half = _in_feats.scalar_type() == at::ScalarType::Half;

  if (is_half)
  {
    // throw std::runtime_error("FP16 kernels have not been updated for split mask implimentation.");
    if (!allow_fp16)
    {
      throw std::runtime_error("FP16 kernels are not supported for implicit GEMM now for SM75-.");
    }
    auto in_feats = reinterpret_cast<half *>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<half *>(_kernel.data_ptr<at::Half>());
    auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 64 / 1;
      dim3 num_blocks(1 * num_in_channels * kernel_volume / 32 * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      conv_backward_cuda_setting2_mode1_f16f16f32<<<num_blocks, threads_per_block>>>(
        _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
    }
    else
    {
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dim3 num_blocks(1 * (num_in_channels + 15) / 16 * kernel_volume * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 1);
      // conv_backward_cuda_setting1_mode1_f16f16f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, false, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 8, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 4, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 2, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 8 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<16, 2, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<8, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<8, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<8, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<8, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<8, 2, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<4, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<4, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<4, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<4, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<4, 2, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<2, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<2, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<2, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<2, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f16f16f32<2, 2, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
    }
  }
  else if (is_tf)
  {
    //throw std::runtime_error("TF32 kernels have not been updated for split mask implimentation.");
    auto in_feats = _in_feats.data_ptr<float>();
    auto kernel = _kernel.data_ptr<float>();
    auto out_feats = _out_feats.data_ptr<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 64 / 1;
      dim3 num_blocks(1 * num_in_channels * kernel_volume / 32 * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      conv_backward_cuda_setting2_mode1_tf32tf32f32<<<num_blocks, threads_per_block>>>(
        _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
    }
    else
    {
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dim3 num_blocks(1 * (num_in_channels + 15) / 16 * kernel_volume * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 1);
      // conv_backward_cuda_setting1_mode1_tf32tf32f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, false, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 8, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 4, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<16, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<8, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_tf32tf32f32<4, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
    }
  }
  else // fp32fp32fp32
  {
    // printf("\nRun FP32 wgrad backward kernels!\n");
    auto in_feats = _in_feats.data_ptr<float>();
    auto kernel = _kernel.data_ptr<float>();
    auto out_feats = _out_feats.data_ptr<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int block_num_M = (num_in_channels * kernel_volume) / 32;
      int block_num_N = (num_out_channels) / 64; //j_factors1

      dim3 num_blocks(block_num_M * block_num_N * split_k_iters);
      dim3 threads_per_block(64);
      conv_backward_cuda_setting2_mode1_f32f32f32<<<num_blocks, threads_per_block>>>(
          _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
    }
    else
    {
      int block_num_M = (num_in_channels + 15) / 16 * kernel_volume;
      int block_num_N = (num_out_channels - 1) / 16 + 1;

      dim3 num_blocks(block_num_M * block_num_N  * split_k_iters);
      dim3 threads_per_block(32);
      // conv_backward_cuda_setting1_mode1_tf32tf32f32<<<num_blocks, threads_per_block>>>(
      //     _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, false, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 8, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 4, false, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<16, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<8, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<8, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<8, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<8, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<4, 16, true, false><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<4, 16, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<4, 8, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode1_f32f32f32<4, 4, true, true><<<num_blocks, threads_per_block>>>(
              _kernel.size(0), num_in_channels, num_out_channels, kernel_volume, split_k_iters, split_mask_len, reduced_mask_len, reorder_loc_len, in_feats, kernel, reduced_mask, out_in_map, reorder_loc, out_feats);
        }
      }
    }
  }
  return _out_feats.sum(0);
}


template <>
torch::Tensor dispatchSparseConvolutionImplicitGEMMGradSorted<torch::kCPU>(
    torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _out_in_map, torch::Tensor _reduced_mask,
    torch::Tensor _reorder_loc, const int split_k_iters,
    bool allow_tf32, bool allow_fp16) {
    TORCH_CHECK(false, "No support for CPU-based ImplicitGEMM!");
}


} // namespace ops
} // namespace detail
} // namespace fvdb
