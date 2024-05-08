import math
from typing import Optional, Union, List, Sequence

import torch
import torch.nn as nn
from torch.profiler import record_function

import fvdb
from fvdb import GridBatch, JaggedTensor
from .vdbtensor import VDBTensor


def fvnn_module(module):
    # Register class as a module in fvdb.nn
    old_forward = module.forward
    def _forward(self, *args, **kwargs):
        with record_function(repr(self)):
            return old_forward(self, *args, **kwargs)
    module.forward = _forward
    return module


GridOrVDBTensor = Union[fvdb.GridBatch, VDBTensor]
ListOrInt = Union[int, List[int]]

@fvnn_module
class MaxPool(nn.Module):
    r"""Applies a 3D max pooling over an input signal.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Note:
        For target voxels that are not covered by any source voxels, the
        output feature will be set to zero.

    """

    def __init__(self, kernel_size: ListOrInt, stride: Optional[ListOrInt] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size

    def forward(self, input: VDBTensor,
                ref_coarse_data: Optional[GridOrVDBTensor] = None) -> VDBTensor:
        if isinstance(ref_coarse_data, VDBTensor):
            coarse_grid, coarse_kmap = ref_coarse_data.grid, ref_coarse_data.kmap
        elif isinstance(ref_coarse_data, fvdb.GridBatch):
            coarse_grid, coarse_kmap = ref_coarse_data, None
        else:
            coarse_grid, coarse_kmap = None, None

        new_feature, new_grid = input.grid.max_pool(
            self.kernel_size, input.feature, stride=self.stride,
            coarse_grid=coarse_grid
        )
        new_feature.jdata[torch.isinf(new_feature.jdata)] = 0.0
        return VDBTensor(new_grid, new_feature, kmap=coarse_kmap)

    def extra_repr(self) -> str:
        return "kernel_size={kernel_size}, stride={stride}".format(
            kernel_size=self.kernel_size, stride=self.stride
        )


@fvnn_module
class AvgPool(nn.Module):
    r"""Applies a 3D average pooling over an input signal.

    Args:
        kernel_size: the size of the window to take average over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    """
    def __init__(self, kernel_size: ListOrInt, stride: Optional[ListOrInt] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or self.kernel_size

    def forward(self, input: VDBTensor,
                ref_coarse_data: Optional[GridOrVDBTensor] = None) -> VDBTensor:
        if isinstance(ref_coarse_data, VDBTensor):
            coarse_grid, coarse_kmap = ref_coarse_data.grid, ref_coarse_data.kmap
        elif isinstance(ref_coarse_data, fvdb.GridBatch):
            coarse_grid, coarse_kmap = ref_coarse_data, None
        else:
            coarse_grid, coarse_kmap = None, None

        new_feature, new_grid = input.grid.avg_pool(
            self.kernel_size, input.feature, stride=self.stride,
            coarse_grid=coarse_grid
        )
        return VDBTensor(new_grid, new_feature, kmap=coarse_kmap)

    def extra_repr(self) -> str:
        return "kernel_size={kernel_size}, stride={stride}".format(
            kernel_size=self.kernel_size, stride=self.stride
        )


@fvnn_module
class UpsamplingNearest(nn.Module):
    r"""Upsamples the input by a given scale factor using nearest upsampling.

    Args:
        scale_factor: the upsampling factor
    """
    def __init__(self, scale_factor: ListOrInt):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(
        self, input: VDBTensor, mask: Optional[JaggedTensor] = None,
        ref_fine_data: Optional[GridOrVDBTensor] = None
    ) -> VDBTensor:
        if isinstance(ref_fine_data, VDBTensor):
            fine_grid, fine_kmap = ref_fine_data.grid, ref_fine_data.kmap
        elif isinstance(ref_fine_data, fvdb.GridBatch):
            fine_grid, fine_kmap = ref_fine_data, None
        else:
            fine_grid, fine_kmap = None, None

        new_feature, new_grid = input.grid.subdivide(
            self.scale_factor, input.feature, mask, fine_grid=fine_grid
        )
        return VDBTensor(new_grid, new_feature, kmap=fine_kmap)

    def extra_repr(self) -> str:
        return "scale_factor={scale_factor}".format(scale_factor=self.scale_factor)


@fvnn_module
class FillToGrid(nn.Module):
    r"""
    Fill the content of input vdb-tensor to another grid.

    Args:
        default_value: the default value to fill in the new grid.
    """
    def __init__(self, default_value: float = 0.0) -> None:
        super().__init__()
        self.default_value = default_value

    def forward(self, input: VDBTensor, other_data: Optional[GridOrVDBTensor] = None) -> VDBTensor:
        if isinstance(other_data, VDBTensor):
            other_grid, other_kmap = other_data.grid, other_data.kmap
        elif isinstance(other_data, fvdb.GridBatch):
            other_grid, other_kmap = other_data, None
        else:
            return input

        new_feature = other_grid.fill_to_grid(input.feature, input.grid, self.default_value)
        return VDBTensor(other_grid, new_feature, kmap=other_kmap)


@fvnn_module
class SparseConv3d(nn.Module):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes, by performing a sparse convolution on the underlying VDB grid.

    Args:
        in_channels: number of channels in the input tensor
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution. Default value is 1
        bias: if ``True``, adds a learnable bias to the output. Default: ``True``
        transposed: if ``True``, uses a transposed convolution operator
    """

    CUTLASS_SUPPORTED_CHANNELS = [
        (32, 64), (64, 128), (128, 256), (32, 32), (64, 64), (128, 128),
        (256, 256), (128, 64), (64, 32), (256, 128), (384, 256), (192, 128),
        (256, 512), (512, 256), (512, 512)
    ]

    """
    Backend for performing convolutions:
      - "default": for now it is 'legacy'
      - "legacy": the old slow implementation
      - "me": MinkowskiEngine implementation
      - "halo": 10x10x10 halo buffer implementation, stride 1, kernel 3, forward only
      - "cutlass": 4x4x6 cutlass implementation, stride 1, kernel 3, forward only, limited channels support
      - "igemm_mode0": unsorted
      - "igemm_mode1": sorted + split=1
      - "igemm_mode2": sorted + split=3
      - "dense": dense convolution
    """
    backend: str = "default"
    allow_tf32: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence] = 3,
        stride: Union[int, Sequence] = 1,
        bias: bool = True,
        transposed: bool = False
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        assert len(kernel_size) == 3

        if isinstance(stride, int):
            stride = (stride,) * 3
        assert len(stride) == 3

        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = transposed

        if self.transposed:
            # Only change kernel size instead of module dict
            out_channels, in_channels = in_channels, out_channels

        self.kernel_volume = math.prod(self.kernel_size)
        if self.kernel_volume > 1:
            # Weight tensor is of shape (Do, Di, K0, K1, K2), but the underlying data is (K2, K1, K0, Di, Do)
            #   so we don't need to make a copy of the permuted tensor within the conv kernel.
            weight_shape = [out_channels, in_channels] + list(self.kernel_size)
            weight = torch.zeros(*weight_shape[::-1]).permute(4, 3, 2, 1, 0)
            self.weight = nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1, 1, 1):
            s += ", stride={stride}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume
        )
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def _dispatch_conv(self, in_feature, in_grid, in_kmap, out_grid):

        backend = self.backend

        if backend == "cutlass" and ((not self.weight.is_cuda) or
                (self.in_channels, self.out_channels) not in self.CUTLASS_SUPPORTED_CHANNELS):
            print(f"Cutlass backend does not support {self.in_channels} -> {self.out_channels} convolutions, falling back to default")
            backend = "default"

        if backend == "default":
            if (not self.weight.is_cuda) or in_feature.dtype == torch.float64:
                backend = "legacy"
            else:
                backend = "igemm_mode1"

        if backend == "halo" and self.stride == (1, 1, 1) and self.kernel_size == (3, 3, 3):
            assert out_grid is None or VDBTensor.same_grid(in_grid, out_grid)
            return in_grid, in_grid.sparse_conv_halo(in_feature, self.weight, 8), None

        elif backend == "dense" and self.stride == (1, 1, 1):
            assert out_grid is None or VDBTensor.same_grid(in_grid, out_grid)
            min_coord = in_grid.ijk.jdata.min(axis=0).values
            # BWHDC -> BCDHW
            dense_feature = in_grid.read_into_dense(in_feature, min_coord=min_coord).permute(0, 4, 3, 2, 1)
            dense_feature = torch.nn.functional.conv3d(
                dense_feature, self.weight, padding=1, stride=1
            )
            # BCDHW -> BWHDC
            dense_feature = dense_feature.permute(0, 4, 3, 2, 1).contiguous()
            dense_feature = in_grid.read_from_dense(dense_feature, dense_origins=min_coord)

            return in_grid, dense_feature, None

        else:
            # Fallback to the default implementation
            can_cache = self.stride == (1, 1, 1) and (out_grid is None or VDBTensor.same_grid(out_grid, in_grid))

            if in_kmap is not None and in_kmap.kernel_size == self.kernel_size and can_cache:
                kmap, out_grid = in_kmap, in_grid
            else:
                if self.transposed:
                    assert out_grid is not None
                    kmap, _ = out_grid.sparse_conv_kernel_map(
                        self.kernel_size, self.stride, in_grid
                    )
                else:
                    kmap, out_grid = in_grid.sparse_conv_kernel_map(
                        self.kernel_size, self.stride, out_grid
                    )

            out_kmap = kmap if can_cache else None

            backend = self._build_kmap_and_convert_backend(kmap, backend)

            if not self.transposed:
                out_feature = kmap.sparse_conv_3d(in_feature, self.weight, backend)
            else:
                out_feature = kmap.sparse_transpose_conv_3d(in_feature, self.weight, backend)

            return out_grid, out_feature, out_kmap

    def _build_kmap_and_convert_backend(self, kmap: fvdb.SparseConvPackInfo, backend: str) -> fvdb.ConvPackBackend:
        if backend in ["legacy", "me"]:
            kmap.build_gather_scatter(backend == "me")
            return fvdb.ConvPackBackend.GATHER_SCATTER

        elif backend == "cutlass":
            kmap.build_cutlass(benchmark=False)
            return fvdb.ConvPackBackend.CUTLASS

        elif backend == "igemm_mode0":
            kmap.build_implicit_gemm(
                sorted=False, split_mask_num=1, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32)
            return fvdb.ConvPackBackend.IGEMM

        elif backend == "igemm_mode1":
            kmap.build_implicit_gemm(
                sorted=True, split_mask_num=1, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32)
            return fvdb.ConvPackBackend.IGEMM

        elif backend == "igemm_mode2":
            kmap.build_implicit_gemm(
                sorted=True, split_mask_num=3, training=self.training, split_mask_num_bwd=3, use_tf32=self.allow_tf32)
            return fvdb.ConvPackBackend.IGEMM

        else:
            raise NotImplementedError(f"Backend {backend} is not supported")

    def forward(
        self,
        input: VDBTensor,
        out_grid: Optional[GridBatch] = None,
    ) -> VDBTensor:
        in_feature, in_grid, in_kmap = input.feature, input.grid, input.kmap

        if self.kernel_size == (1, 1, 1) and self.stride == (1, 1, 1):
            out_feature = in_feature.jdata.matmul(self.weight.transpose(0, 1))
            out_feature = in_feature.jagged_like(out_feature)
            out_grid, out_kmap = in_grid, in_kmap

        else:
            out_grid, out_feature, out_kmap = self._dispatch_conv(
                in_feature, in_grid, in_kmap, out_grid
            )

        if self.bias is not None:
            out_feature.jdata = out_feature.jdata + self.bias

        return VDBTensor(out_grid, out_feature, out_kmap)


@fvnn_module
class GroupNorm(nn.GroupNorm):
    r"""Applies Group Normalization over a VDBTensor.
    See :class:`~torch.nn.GroupNorm` for detailed information.
    """
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.feature.jdata.size(1)
        assert num_channels == self.num_channels, "Input feature should have the same number of channels as GroupNorm"
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.feature.jdata, input.feature.joffsets

        result_data = torch.empty_like(flat_data)

        for b in range(num_batches):
            feat = flat_data[flat_offsets[b, 0]:flat_offsets[b, 1]]
            if feat.size(0) != 0:
                feat = feat.transpose(0, 1).reshape(1, num_channels, -1)
                feat = super().forward(feat)
                feat = feat.reshape(num_channels, -1).transpose(0, 1)

                result_data[flat_offsets[b, 0]:flat_offsets[b, 1]] = feat

        return VDBTensor(input.grid, input.grid.jagged_like(result_data), input.kmap)


@fvnn_module
class BatchNorm(nn.BatchNorm1d):
    r"""Applies Batch Normalization over a VDBTensor.
    See :class:`~torch.nn.BatchNorm1d` for detailed information.
    """
    def forward(self, input: VDBTensor) -> VDBTensor:
        num_channels = input.feature.jdata.size(1)
        assert num_channels == self.num_features, "Input feature should have the same number of channels as BatchNorm"
        result_data = super().forward(input.feature.jdata)
        return VDBTensor(input.grid, input.grid.jagged_like(result_data), input.kmap)


@fvnn_module
class ElementwiseMixin:
    def forward(self, input: VDBTensor) -> VDBTensor:
        assert isinstance(input, VDBTensor), "Input should have type VDBTensor"
        res = super().forward(input.feature.jdata) # type: ignore
        return VDBTensor(input.grid, input.feature.jagged_like(res), input.kmap)


class Linear(ElementwiseMixin, nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    """


class ReLU(ElementwiseMixin, nn.ReLU):
    r"""
    Applies the rectified linear unit function element-wise: :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """


class LeakyReLU(ElementwiseMixin, nn.LeakyReLU):
    r"""
    Applies the element-wise function: :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    """


class SELU(ElementwiseMixin, nn.SELU):
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = \lambda \left\{
    \begin{array}{lr}
    x, & \text{if } x > 0 \\
    \text{negative\_slope} \times e^x - \text{negative\_slope}, & \text{otherwise }
    \end{array}
    \right.`
    """


class SiLU(ElementwiseMixin, nn.SiLU):
    r"""
    Applies element-wise, :math:`\text{SiLU}(x) = x * \sigma(x)`, where :math:`\sigma(x)` is the sigmoid function.
    """


class Tanh(ElementwiseMixin, nn.Tanh):
    r"""
    Applies element-wise, :math:`\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}`
    """


class Sigmoid(ElementwiseMixin, nn.Sigmoid):
    r"""
    Applies element-wise, :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """


class Dropout(ElementwiseMixin, nn.Dropout):
    r"""
    During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`
    using samples from a Bernoulli distribution. The elements to zero are randomized on every forward call.
    """
