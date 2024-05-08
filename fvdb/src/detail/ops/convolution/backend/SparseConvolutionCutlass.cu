#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/convolution/backend/ConvOps.h"

namespace example {

using namespace cute;

// Empty type used to disable gather/scatter for a GEMM argument
struct NoGather
{
  template<class... Ts>
  NoGather(Ts...) {};
};

/// Function object that applies an index to its argument
template <class Index>
struct IndexedGather
{
  CUTE_HOST_DEVICE constexpr
  IndexedGather(Index const *indices = {}): indices_(indices) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr
  Index
  operator()(I i) const { return indices_[i]; }

  CUTE_HOST_DEVICE friend
  void
  print(IndexedGather const &s) {
    print("Indexed");
  }

  Index const *indices_;
};

/// Function object that applies a stride to its argument
/// Example: StridedFunc<int,_2> gathers every other row/column
template <class Stride>
struct StridedGather
{
  CUTE_HOST_DEVICE constexpr
  StridedGather(Stride stride = {}): stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(I i) const { return i * stride_; }

  CUTE_HOST_DEVICE friend
  void
  print(StridedGather const &s) {
    print("Strided{");
    print(s.stride_);
    print("}");
  }

  Stride stride_;
};

/// Custom stride object that applies a function followed by a stride
template <class Func, class Stride>
struct CustomStride
{
  CUTE_HOST_DEVICE constexpr
  CustomStride(Func const &func, Stride const &stride): func_(func), stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr friend
  auto
  operator*(I i, CustomStride const &s) { return s.func_(i) * s.stride_; }

  CUTE_HOST_DEVICE friend
  void
  print(CustomStride const & s) {
    print("Custom{");
    print(s.func_);
    print(",");
    print(s.stride_);
    print("}");
  }

  template<class Div>
  CUTE_HOST_DEVICE constexpr friend
  auto
  safe_div(CustomStride const &s, Div const &div)
  {
    return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_, safe_div(s.stride_, div));
  }

  // Circumvent the requirement on make_layout that shape and stride are integral
  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend
  auto
  make_layout(Shape const &shape, CustomStride const &stride)
  {
    return Layout<Shape, CustomStride>(shape, stride);
  }

  Func func_;
  Stride stride_;
};

template<class Stride, class Func>
CUTLASS_HOST_DEVICE
auto
make_custom_stride_layout(Stride const &stride, Func&& func)
{
  // Use a dummy shape and replace the first non-unit stride with a custom gather stride
  auto idx = find_if(stride, [](auto x){ return not is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(repeat_like(stride, _1{}),
                     replace<I>(stride, CustomStride{static_cast<Func&&>(func), get<I>(stride)}));
}

/// Helper function to optionally create a gather tensor
template<class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE
auto
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func)
{
  if constexpr (not cutlass::platform::is_same<remove_cvref_t<Func>, NoGather>::value) {
    Layout matrix_layout = make_identity_layout(shape);
    auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
    Layout gather_layout = make_custom_stride_layout(stride, static_cast<Func&&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
  } else {
    return make_tensor(iter, shape, stride);
  }
}

} // namespace example

namespace cute
{

template<int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(Shape const& shape, Stride const& stride)
{
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) { return upcast<N,I>(s,d); });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(shape_div(shape, Int<N>{}), shape_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<Layout<OuterShape,OuterStride>,Offset,Layout<Shape,Stride>> const& layout)
{
  // Find index of the stride-1 mode - that is the only one that requires updating inner shape and offset
  auto idx = find_if(layout.layout_a().stride(), [](auto x){ return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset = as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner = upcast<N,I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

} // namespace example

namespace fvdb {
namespace detail {
namespace ops {

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || __cplusplus >= 202002L)

using namespace cute;
using example::IndexedGather;
using example::CustomStride;

template <typename IntDi, typename IntDo>
struct KernelFunctorV1 {
    //
    // Static config
    //
    using D = _6;
    using H = _4;
    using W = _4;

    using T = _3;
    using R = _3;
    using S = _3;

    using Z = _4;
    using P = _2;
    using Q = _2;

    using C = IntDi;
    using K = IntDo;

    using Tiler_K = decltype(cute::min(K{}, _128{}));;
    using Tiler_C = decltype(cute::min(C{}, _32{}));
    using Tiler_N = _4;
    using TileM = Tiler_K;
    using TileN = Shape<Tiler_N, Z, P, Q>;
    using TileK = Shape<Tiler_C,_1,_1,_1>;
    using PIPE  = _3;
    using TilerFlt = Shape<TileM, TileK>;
    using TilerAct = Shape<TileN, TileK>;
    using TilerOut = Shape<TileM, TileN>;

    using TileSizeM = Int<size(TileM{})>;
    using TileSizeN = Int<size(TileN{})>;
    using TileSizeK = Int<size(TileK{})>;
    static constexpr int Stages = PIPE::value;

    // TODO: add rounding if input types are fp32 instead of tf32
    using ElementFlt = tfloat32_t;
    using ElementAct = tfloat32_t;
    using ElementOut = float;

    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2,_2,_1>>,
        Tile<_32,_32,Underscore>>;

    static constexpr int MaxThreadsPerBlock = size(TiledMma{});
    static constexpr int MinBlocksPerMultiprocessor = 1;

    union SharedStorage {
        struct {
            ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
            ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
        } mainloop;

        struct {
            ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
        } epilogue;
    };

    //
    // Stencil tensor
    //

    using GmemLayoutFlt = decltype(make_ordered_layout(
        Shape< K, Shape< C, T, R, S>>{},
        tuple<_4, tuple<_0,_3,_2,_1>>{}));

    // We have 64 elements * 32b each in the major mode that we can vectorize
    // Max vector size is 128b, so lay 16 threads along the major mode with a vector size of 4
    // Rest along the minor mode
    using GmemTiledCopyFlt = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementFlt>{},
        Layout<Shape <_16, _8>,
               Stride< _8, _1>>{},
        Layout<Shape < _1, _4>>{}));

    // using SmemLayoutFlt = decltype(
    //     composition(Swizzle<3,2,3>{},
    //                 make_ordered_layout(
    //                     Shape<TileSizeM,TileSizeK,PIPE>{},
    //                     tuple<       _1,       _0,  _2>{})));

    using SmemLayoutFlt = decltype(
        tile_to_shape(
            composition(Swizzle<1,2,3>{},
                        Layout<Shape <_8,Shape <_4, _2>>,
                               Stride<_4,Stride<_1,_32>>>{}),
            Shape<TileSizeM,TileSizeK,PIPE>{}));

    using SmemCopyAtomFlt = Copy_Atom<SM75_U32x4_LDSM_N, ElementFlt>;

    //
    // Activation tensor
    //

    // Activation tensor is major in the contraction mode, so vectorize that mode first
    // Then lay out the rest of the threads along the other mode
    using GmemTiledCopyAct = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementAct>{},
        Layout<Shape <_16, _8>,
               Stride< _8, _1>>{},
        Layout<Shape < _1, _4>>{}));

    // Both Flt and Act are contraction major
    // using SmemLayoutAct = decltype(
    //     composition(Swizzle<3,2,3>{},
    //                 make_ordered_layout(
    //                     Shape<TileSizeN,TileSizeK,PIPE>{},
    //                     tuple<       _1,       _0,  _2>{})));

    using SmemLayoutAct = decltype(
        tile_to_shape(
            composition(Swizzle<1,2,3>{},
                        Layout<Shape <_8,Shape <_4, _2>>,
                               Stride<_4,Stride<_1,_32>>>{}),
            Shape<TileSizeN,TileSizeK,PIPE>{}));

    using SmemCopyAtomAct = Copy_Atom<SM75_U32x4_LDSM_N, ElementAct>;

    //
    // Output tensor
    //

    using GmemTiledCopyOut = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, ElementAct>{},
        Layout<Shape <_8, _16>,
               Stride<_1,  _8>>{},
        Layout<Shape <_4,  _1>>{}));

    using SmemCopyAtomOut = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

    // TODO: this is not the best swizzle ...
    using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

    //
    // Conv functor
    //
    template <class EngineFlt, class TensorActivation, class TensorOutput>
    void __device__
    operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt, // ( K,        (C,T,R,S))
               TensorActivation                       mAct, // ((N,Z,P,Q), (C,T,R,S))
               TensorOutput                           mOut, // ( K,        (N,Z,P,Q))
               char* smem_buf) const {
        using namespace cute;
        uint64_t start = clock64();
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        Tensor sA = make_tensor(make_smem_ptr(&storage.mainloop.sAMatrix[0]), SmemLayoutFlt{});
        Tensor sB = make_tensor(make_smem_ptr(&storage.mainloop.sBMatrix[0]), SmemLayoutAct{});
        Tensor sC = make_tensor(make_smem_ptr(&storage.epilogue.sCMatrix[0]), SmemLayoutOut{});

        TiledMma tiled_mma;
        Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});
        clear(accum);
        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

        // Set up tensors
        // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of higher dynamic range in NDHW
        Tensor gA_mk = local_tile(mFlt, TilerFlt{}, make_coord(_,_));                              // (BLK_M,BLK_K,m',k')
        Tensor gB_nk = local_tile(mAct, TilerAct{}, make_coord(_,_));                              // (BLK_N,BLK_K,n',_1)
        Tensor gC_mn = local_tile(mOut, TilerOut{}, make_coord(_,_));                              // (BLK_M,BLK_N,m',n')

        // Compute m_coord and n_coord with their post-tiled shapes
        auto m_coord = idx2crd(int(blockIdx.y), shape<2>(gA_mk));
        auto n_coord = idx2crd(int(blockIdx.x), shape<2>(gB_nk));
        Tensor gA = gA_mk(_,_,m_coord,_);                                                          // (BLK_M,BLK_K,k')
        Tensor gB = gB_nk(_,_,n_coord,_);                                                          // (BLK_N,BLK_K,_1)
        Tensor gC = gC_mn(_,_,m_coord,n_coord);                                                    // (BLK_M,BLK_N)

        GmemTiledCopyFlt gmem_tiled_copy_A;
        auto gmem_thr_copy_A   = gmem_tiled_copy_A.get_slice(threadIdx.x);
        Tensor tAgA            = gmem_thr_copy_A.partition_S(gA);                                  // (VEC,ACPY_M,ACPY_K,k')
        Tensor tAsA            = gmem_thr_copy_A.partition_D(sA);                                  // (VEC,ACPY_M,ACPY_K,PIPE)

        GmemTiledCopyAct gmem_tiled_copy_B;
        auto gmem_thr_copy_B   = gmem_tiled_copy_B.get_slice(threadIdx.x);
        Tensor tBgB            = gmem_thr_copy_B.partition_S(gB);                                  // (VEC,ACPY_N,ACPY_K,_1)
        Tensor tBsB            = gmem_thr_copy_B.partition_D(sB);                                  // (VEC,ACPY_N,ACPY_K,PIPE)

        // Copy and MMA partitioning
        Tensor tCrA            = thr_mma.partition_fragment_A(sA(_,_,Int<0>{}));                   // (VEC,MMA_M,MMA_K)
        Tensor tCrB            = thr_mma.partition_fragment_B(sB(_,_,Int<0>{}));                   // (VEC,MMA_N,MMA_K)

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomFlt{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
        Tensor tCsA            = smem_thr_copy_A.partition_S(sA);                                  // (VEC,CPY_M,CPY_K,PIPE)
        Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomAct{}, tiled_mma);
        auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
        Tensor tCsB            = smem_thr_copy_B.partition_S(sB);                                  // (VEC,CPY_N,CPY_K,PIPE)
        Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

        //
        // Prologue
        //
        int k_tile_count = size<2>(gA);
        // XXX: should be multimode (C/TILER_C, (S,R,T)) but I am going to shave off a few k clks and flatten this for now
        auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
        constexpr int K_BLOCK_MAX = size<2>(tCrA); // Size of the register pipeline

        static_assert(Stages >= 2);
        static_assert(K_BLOCK_MAX > 1);

        // Current pipe index in smem for mma to read from
        int smem_pipe_read  = 0;
        // Current pipe index in smem for gmem to write to
        int smem_pipe_write = 0;

        // ramp up the gmem->smem load pipeline
        CUTE_UNROLL
        for (; smem_pipe_write < Stages-1; ++smem_pipe_write) {
            copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
            copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_iter; }
        }

        // ramp up the smem->rmem load pipeline
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<Stages - 2>();
        __syncthreads();

        // uint64_t prologue_gmem_2_smem = clock64();

        // Prefetch the first rmem fragment for filter
        Tensor tCsA_read = tCsA(_,_,_,smem_pipe_read);
        copy(smem_tiled_copy_A, tCsA_read(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        Tensor tCsB_read = tCsB(_,_,_,smem_pipe_read);
        copy(smem_tiled_copy_B, tCsB_read(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));

        //
        // Mainloop
        //
        // uint64_t mainloop_start = clock64();

        // XXX: WARNING this loop does not support predication over any mode!
        CUTE_NO_UNROLL
        for (; k_tile_count > -(Stages-1); --k_tile_count) { // trip count = S*R*T*(C / Tiler_C)
            // Pipeline the outer products with a static for loop.
            for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    // Commit the smem for smem_pipe_read
                    cp_async_wait<Stages - 2>();
                    __syncthreads();
                    // Advance the smem->rmem pipeline
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
                    tCsA_read = tCsA(_,_,_,smem_pipe_read);
                    tCsB_read = tCsB(_,_,_,smem_pipe_read);
                }

                // Load smem->rmem for k_block+1
                constexpr auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
                copy(smem_tiled_copy_A, tCsA_read(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
                copy(smem_tiled_copy_B, tCsB_read(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

                // Copy gmem to smem before computing gemm on each k-pipe
                if (k_block == 0) {
                    copy(gmem_tiled_copy_A, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
                    copy(gmem_tiled_copy_B, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
                    cp_async_fence();
                    // Advance the gmem->smem pipeline
                    if (k_tile_count > 0) { ++k_tile_iter; }
                    smem_pipe_write = smem_pipe_read;
                }

                // gemm for k_block
                cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
            });
        }

        cp_async_wait<0>();
        __syncthreads();

        //
        // Epilogue
        //
        // uint64_t epi_start = clock64();

        auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomOut{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_slice(threadIdx.x);
        auto tCrC = smem_thr_copy_C.retile_S(accum);
        auto tCsC = smem_thr_copy_C.partition_D(sC);
        copy(smem_tiled_copy_C, tCrC, tCsC);

        __syncthreads();

        GmemTiledCopyOut gmem_tiled_copy_C;
        auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
        auto tDsC = gmem_thr_copy_C.partition_S(sC);
        auto tDgC = gmem_thr_copy_C.partition_D(gC);
        copy(gmem_tiled_copy_C, tDsC, tDgC);

        // uint64_t end = clock64();
    }
};

template<class Operator, class FilterTensor, class ActivationTensor, class OutputTensor>
__global__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
void kernel_entrypoint(FilterTensor mFlt, ActivationTensor mAct, OutputTensor mOut) {
    extern __shared__ char smem_buf[];
    Operator op;
    op(mFlt, mAct, mOut, smem_buf);
}

template <int Di, int Do>
int stencilConvolveLauncher(
        size_t num_bricks,
        uint32_t *halo_index_buffer,
        float *inputBuffer,
        float *stencil,
        float *outputBuffer,
        uint32_t *output_index_buffer) {

    using KernelFunctor = KernelFunctorV1<cute::Int<Di>, cute::Int<Do>>;

    using D_t = typename KernelFunctor::D;
    using H_t = typename KernelFunctor::H;
    using W_t = typename KernelFunctor::W;
    using Z_t = typename KernelFunctor::Z;
    using P_t = typename KernelFunctor::P;
    using Q_t = typename KernelFunctor::Q;
    using C_t = typename KernelFunctor::C;
    using K_t = typename KernelFunctor::K;
    using S_t = typename KernelFunctor::S;
    using R_t = typename KernelFunctor::R;
    using T_t = typename KernelFunctor::T;

    int N = num_bricks; // dynamic
    if (N % int(typename KernelFunctor::Tiler_N{}) != 0) {
        printf("ERROR: Input image count must be evenly divisible by CTA tiler N. Got num_bricks = %d\n", N);
        return 1;
    }

    auto D = D_t{};
    auto H = H_t{};
    auto W = W_t{};
    auto Z = Z_t{};
    auto P = P_t{};
    auto Q = Q_t{};
    auto C = C_t{};
    auto K = K_t{};
    auto S = S_t{};
    auto R = R_t{};
    auto T = T_t{};

    // Tensor Filter    : (k,c,s,r,t)::(128,3,3,3,64):(1728,576,192,64,1)
    // auto filter_layout = KernelFunctor::GmemLayoutFlt();
    typename KernelFunctor::GmemLayoutFlt filter_layout{};

    // Tensor Output    : (n,z,p,q,k)::(?,4,2,2,128):(2048,1024,512,128,1)
    auto output_layout = make_ordered_layout(
        make_shape( K,   make_shape( N,   Z,   P,   Q)),
        make_tuple(_0{}, make_tuple(_4{},_3{},_2{},_1{})));

    // Input gather layout
    auto xformed_act_layout = make_layout(
        make_shape (make_shape(N, Z, P, Q), make_shape (C, T, R, S)),
        make_stride(make_stride(C*D*H*W, H*W*C, C*W, C), make_stride(_1{}, H*W*C, C*W, C))); // -> idx

    auto xformed_act_shape = shape(xformed_act_layout);
    auto xformed_act_basis_stride = make_stride(
        make_stride(W*H*D*E<0>{}, W*H*E<0>{}, W*E<0>{}, _1{}*E<0>{}),
        make_stride(      E<1>{}, W*H*E<0>{}, W*E<0>{}, _1{}*E<0>{}));

    // gather_tensor_layout(make_coord((nzpq), (csrt))) => (idx_buffer_idx, dense_c_idx)
    auto xformed_act_basis_layout = make_layout(xformed_act_shape, xformed_act_basis_stride);
    // gather_tensor_layout(make_coord(idx_buffer_idx, dense_c_idx)) => idx in input values buffer
    auto xformed_act_gather_layout = make_layout(
        make_shape(_1{},_1{}),
        make_stride(CustomStride{IndexedGather{halo_index_buffer}, C}, _1{}));

    // Composed layout that takes the composes the idx buf index and c index to map to values
    // ((nzpq), (csrt)) => (idx_buffer_idx, dense_c_idx) => (gmem_base_ptr + halo_index_buf[idx_buffer_idx])[dense_c_idx]
    // XXX: CustomStride is scaling the loaded index value by the C dimension, so our offset vector should be unscaled
    auto xformed_act_composed_layout = composition(
        xformed_act_gather_layout,
        make_arithmetic_tuple(_0{}, _0{}),
        xformed_act_basis_layout);

    // Output scatter layout
    auto out_basis_stride = make_stride(
        E<1>{},
        make_stride(Z*P*Q*E<0>{}, P*Q*E<0>{}, Q*E<0>{}, _1{}*E<0>{})); // -> (crd0, crd1)
    auto out_basis_layout = make_layout(shape(output_layout), out_basis_stride);
    auto out_scatter_layout = make_layout(
        make_shape(_1{},_1{}),
        make_stride(CustomStride{IndexedGather{output_index_buffer}, K}, _1{}));
    auto out_composed_layout = composition(
        out_scatter_layout,
        make_arithmetic_tuple(_0{},_0{}),
        out_basis_layout);

    cute::Tensor mXformedActGather = make_tensor(make_gmem_ptr(inputBuffer), xformed_act_composed_layout);
    cute::Tensor mFilter = make_tensor(make_gmem_ptr(stencil), filter_layout);
    cute::Tensor mOutputScatter = make_tensor(make_gmem_ptr(outputBuffer), out_composed_layout);  // (K, (N,Z,P,Q))

    cute::Tensor gOutput_mn = zipped_divide(mOutputScatter, typename KernelFunctor::TilerOut{}); // ((BLK_M, BLK_N), (m', n'))
    dim3 lauch_grid {size<1,1>(gOutput_mn), size<1,0>(gOutput_mn), 1};
    constexpr size_t smem_size = sizeof(typename KernelFunctor::SharedStorage);

    #if 0
        print("xforemed gather layout ((N,Z,P,Q), (C,T,R,S)) = "); print(xformed_act_composed_layout); print("\n");
        print("Output          layout ( K,        (N,Z,P,Q)) = "); print(output_layout);               print("\n");
        print("Output  scatter layout ( K,        (N,Z,P,Q)) = "); print(out_composed_layout);         print("\n");
        print("Filter layout          ( K,        (C,T,R,S)) = "); print(filter_layout);               print("\n");
        print("Tiled Output layout = "); print(gOutput_mn.layout());                                   print("\n");
    #endif
    #if 0
        for (int n = 0; n < 1; ++n)
        for (int z = 0; z < size<0,1>(mXformedActGather); ++z)
        for (int p = 0; p < size<0,2>(mXformedActGather); ++p)
        for (int q = 0; q < size<0,3>(mXformedActGather); ++q) {
            auto coord_out = make_coord(_0{}, make_coord(n,z,p,q));
            auto out_idx_channeloffset_pair = out_basis_layout(coord_out);
            auto out_idx = out_scatter_layout(out_idx_channeloffset_pair);
            auto out_idx_2 = out_composed_layout(coord_out);
            print("out_basis_layout"); print(coord_out);
                print(" => idx_offset pair "); print(out_idx_channeloffset_pair);
                print(" => scatter idx "); print(out_idx); print(" | ");print(out_idx_2); print("\n");

            for (int s = 0; s < size<1,1>(mXformedActGather); ++s)
            for (int r = 0; r < size<1,2>(mXformedActGather); ++r)
            for (int t = 0; t < size<1,3>(mXformedActGather); ++t)
            for (int c = 0; c < size<1,0>(mXformedActGather); ++c) {
                auto coord = make_coord(make_coord(n,z,p,q), make_coord(c,s,r,t));
                auto idx_channeloffset_pair = xformed_act_basis_layout(coord);
                auto input_idx = xformed_act_gather_layout(idx_channeloffset_pair);
                auto input_idx_2 = xformed_act_composed_layout(coord);
                float val_at_idx = mXformedActGather(coord);
                assert(input_idx == input_idx_2);
                print("\txformed_act_basis_layout"); print(coord);
                    print(" => idx_offset pair "); print(idx_channeloffset_pair);
                    print(" => gather idx "); print(input_idx_2);
                    print(" => input value "); print(val_at_idx); print("\n");
            }
        }
    #endif

    cudaFuncSetAttribute(
        kernel_entrypoint<KernelFunctor, decltype(mFilter), decltype(mXformedActGather), decltype(mOutputScatter)>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    kernel_entrypoint<KernelFunctor, decltype(mFilter), decltype(mXformedActGather), decltype(mOutputScatter)>
        <<<lauch_grid, KernelFunctor::MaxThreadsPerBlock, smem_size>>>(
            mFilter, mXformedActGather, mOutputScatter);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // milliseconds /= 1;

    #if 0
    double tflop_count = (2 * double(size<0>(xformed_act_shape)) * double(size(filter_layout))) / double(1e12);
    double tflops = tflop_count / (double(milliseconds) / double(1e3));
    printf("Conv TFLOP count = %f\n", tflop_count);
    printf("GPU convolution: %fms. TFLOP/s = %f\n", milliseconds, tflops);
    #endif

    return 0;
}


template <>
torch::Tensor dispatchSparseConvolutionCutlass<torch::kCUDA>(
        const torch::Tensor& inFeatures, const torch::Tensor& kernel,
        const torch::Tensor& haloIndexBuffer, const torch::Tensor& outputIndexBuffer,
        bool benchmark) {

    // Assuming kernel is reshaped from [Do, Di, D, H, W] to [Do, D, H, W, Di] outside
    const int inC = kernel.size(4), outC = kernel.size(0);

    torch::Tensor paddedInFeatures;
    torch::Tensor outFeatures;

    if (!benchmark) {
      // Pre-pad input features with 0
      paddedInFeatures = torch::zeros({inFeatures.size(0) + 1, inC}, inFeatures.options());
      paddedInFeatures.slice(0, 1, inFeatures.size(0) + 1) = inFeatures;
      outFeatures = torch::zeros({paddedInFeatures.size(0), outC}, inFeatures.options());
    } else {
      paddedInFeatures = inFeatures;
      outFeatures = torch::empty({inFeatures.size(0), outC}, inFeatures.options());
    }

    // Run convolution
    auto convFunc = stencilConvolveLauncher<64, 128>;

    if (inC == 32 && outC == 64) {
      convFunc = stencilConvolveLauncher<32, 64>;
    } else if (inC == 64 && outC == 128) {
      convFunc = stencilConvolveLauncher<64, 128>;
    } else if (inC == 128 && outC == 256) {
      convFunc = stencilConvolveLauncher<128, 256>;
    } else if (inC == 32 && outC == 32) {
      convFunc = stencilConvolveLauncher<32, 32>;
    } else if (inC == 64 && outC == 64) {
      convFunc = stencilConvolveLauncher<64, 64>;
    } else if (inC == 128 && outC == 128) {
      convFunc = stencilConvolveLauncher<128, 128>;
    } else if (inC == 128 && outC == 64) {
      convFunc = stencilConvolveLauncher<128, 64>;
    } else if (inC == 256 && outC == 256) {
      convFunc = stencilConvolveLauncher<256, 256>;
    } else if (inC == 256 && outC == 128) {
      convFunc = stencilConvolveLauncher<256, 128>;
    } else if (inC == 64 && outC == 32) {
      convFunc = stencilConvolveLauncher<64, 32>;
    } else if (inC == 384 && outC == 256) {
      convFunc = stencilConvolveLauncher<384, 256>;
    } else if (inC == 192 && outC == 128) {
      convFunc = stencilConvolveLauncher<192, 128>;
    } else if (inC == 512 && outC == 512) {
      convFunc = stencilConvolveLauncher<512, 512>;
    } else if (inC == 512 && outC == 256) {
      convFunc = stencilConvolveLauncher<512, 256>;
    } else if (inC == 256 && outC == 512) {
      convFunc = stencilConvolveLauncher<256, 512>;
    } else {
      TORCH_CHECK(false, "Unsupported convolution size, inC = " + std::to_string(inC) + ", outC = " + std::to_string(outC));
    }

    convFunc(
        haloIndexBuffer.size(0) / 96,
        (uint32_t *) haloIndexBuffer.data_ptr<int>(),
        paddedInFeatures.data_ptr<float>(),
        kernel.data_ptr<float>(),
        outFeatures.data_ptr<float>(),
        (uint32_t *) outputIndexBuffer.data_ptr<int>());

    if (!benchmark) {
      // Slice out the padded part
      outFeatures = outFeatures.slice(0, 1, inFeatures.size(0) + 1);
    }

    return outFeatures;
}

#else

template <>
torch::Tensor dispatchSparseConvolutionCutlass<torch::kCUDA>(
        const torch::Tensor& inFeatures, const torch::Tensor& kernel,
        const torch::Tensor& haloIndexBuffer, const torch::Tensor& outputIndexBuffer,
        bool benchmark) {

    TORCH_CHECK(false, "CUDA <= 12.0 does not support c++20 standard. Compile with a newer nvcc.");
}

#endif


template <>
torch::Tensor dispatchSparseConvolutionCutlass<torch::kCPU>(
        const torch::Tensor& inFeatures, const torch::Tensor& kernel,
        const torch::Tensor& haloIndexBuffer, const torch::Tensor& outputIndexBuffer,
        bool benchmark) {
    TORCH_CHECK(false, "CPU not supported for SparseConvolutionHalo yet!");
}


} // namespace ops
} // namespace detail
} // namespace fvdb

