import functools
import itertools
import unittest
from typing import List

import torch
import torch.nn as nn
from parameterized import parameterized

import fvdb
import fvdb.nn as fvnn
from fvdb import GridBatch

from .common import test_expand

all_device_dtype_combos = [
    ['cuda', torch.float16, False],
    ['cpu', torch.float32, False],
    ['cuda', torch.float32, False],
    ['cpu', torch.float64, False],
    ['cuda', torch.float64, False],
    ['cuda', torch.float16, True],
    ['cpu', torch.float32, True],
    ['cuda', torch.float32, True],
    ['cpu', torch.float64, True],
    ['cuda', torch.float64, True]
]


def dtype_to_atol(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 1e-1
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float64:
        return 1e-5
    raise ValueError("dtype must be a valid torch floating type")


def get_module(module_name: str, is_fvdb: bool = False):
    match module_name:
        case 'Conv':
            return fvnn.SparseConv3d if is_fvdb else functools.partial(nn.Conv3d, padding=1)
        case 'MaxPool':
            return fvnn.MaxPool if is_fvdb else nn.MaxPool3d
        case 'GroupNorm':
            return fvnn.GroupNorm if is_fvdb else nn.GroupNorm
        case 'ReLU':
            return fvnn.ReLU if is_fvdb else nn.ReLU
        case 'UpsamplingNearest':
            return fvnn.UpsamplingNearest if is_fvdb else nn.Upsample
        case _:
            raise NotImplementedError


class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 kernel_size: int = 3,
                 is_fvdb: bool = False):
        super().__init__()
        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', get_module('ReLU', is_fvdb)(inplace=True)) # type: ignore
            elif char == 'c':
                self.add_module('Conv', get_module('Conv', is_fvdb)(
                    in_channels, out_channels, kernel_size, 1, bias='g' not in order)) # type: ignore
            elif char == 'g':
                if i < order.index('c'):
                    num_channels = in_channels
                else:
                    num_channels = out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module('GroupNorm', get_module('GroupNorm', is_fvdb)(
                    num_groups=num_groups, num_channels=num_channels, affine=False)) # type: ignore
            else:
                raise NotImplementedError


class SparseDoubleConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 num_groups: int,
                 encoder: bool,
                 is_fvdb: bool = False):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            self.add_module('MaxPool', get_module('MaxPool', is_fvdb)(kernel_size=2)) # type: ignore
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1',
                        ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups, is_fvdb=is_fvdb))
        self.add_module('SingleConv2',
                        ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups, is_fvdb=is_fvdb))


class MyUNet(nn.Module):
    def __init__(self, is_fvdb: bool = False):
        super().__init__()
        self.is_fvdb = is_fvdb
        self.enc0 = SparseDoubleConv(3, 64, 'gcr', 8, True, is_fvdb=is_fvdb)
        self.enc1 = SparseDoubleConv(64, 128, 'gcr', 8, True, is_fvdb=is_fvdb)
        self.enc2 = SparseDoubleConv(128, 256, 'gcr', 8, True, is_fvdb=is_fvdb)
        self.dec2 = SparseDoubleConv(256, 128, 'gcr', 8, False, is_fvdb=is_fvdb)
        self.dec1 = SparseDoubleConv(128, 64, 'gcr', 8, False, is_fvdb=is_fvdb)
        self.dec0 = SparseDoubleConv(64, 1, 'gcr', 8, False, is_fvdb=is_fvdb)
        self.upsample = get_module('UpsamplingNearest', is_fvdb)(scale_factor=2) # type: ignore

    def forward(self, x_0):
        x_1 = self.enc0(x_0)
        x_2 = self.enc1(x_1)
        x_3 = self.enc2(x_2)
        y_2 = self.upsample(x_3)
        y_2 = self.dec2(y_2)
        y_1 = self.upsample(y_2)
        y_1 = self.dec1(y_1)
        y_0 = self.upsample(y_1)
        y_0 = self.dec0(y_0)
        return y_0


def copy_weights(dst_model: nn.Module, src_model: nn.Module):
    for dst_name, dst_param in dst_model.named_parameters():
        src_param = src_model.state_dict()[dst_name]
        dst_param.data.copy_(src_param)


class TestNN(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_dtype_combos)
    def test_group_norm(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3

        for affine in (True, False):
            for num_channels in (64, 128):
                for num_groups in (8, 16, 32, 64):
                    vdb_features = grid.jagged_like(
                        torch.randn((grid.total_voxels, num_channels),
                                    device=device, dtype=dtype, requires_grad=True))

                    # Our groupnorm
                    our_gn_op = fvnn.GroupNorm(
                        num_groups=num_groups, num_channels=num_channels, dtype=dtype, affine=affine).to(device)
                    our_gn_input = fvnn.VDBTensor(grid, vdb_features)
                    our_gn_output = our_gn_op(our_gn_input)
                    our_gn_output.feature.jdata.sum().backward()
                    our_gn_grad = our_gn_input.feature.jdata.grad.clone()

                    # Pytorch groupnorm
                    torch_gn_op = torch.nn.GroupNorm(
                        num_groups=num_groups, num_channels=num_channels, dtype=dtype, affine=affine).to(device)
                    torch_gn_input = vdb_features.jdata.view(1, gsize, gsize, gsize, num_channels).permute(0, 4, 1, 2, 3)
                    torch_gn_input = torch_gn_input.detach().clone().requires_grad_(True)
                    torch_gn_output = torch_gn_op(torch_gn_input).permute(0, 2, 3, 4, 1).reshape(-1, num_channels)
                    torch_gn_output.sum().backward()
                    assert torch_gn_input.grad is not None
                    torch_gn_grad = torch_gn_input.grad.clone().reshape(-1, num_channels)

                    # Check that the outputs are the same
                    self.assertTrue(torch.mean(our_gn_output.feature.jdata - torch_gn_output) < 1e-3)
                    self.assertTrue(torch.mean(our_gn_grad - torch_gn_grad) < 1e-3)

    @parameterized.expand(all_device_dtype_combos)
    def test_max_pool(self, device, dtype, mutable):
        vox_size = 0.05
        vox_origin = (0.0, 0.0, 0.0)
        gsize = int(1 / vox_size)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, [20, 20, 20], voxel_sizes=vox_size, origins=vox_origin)
        assert grid.total_voxels == 20 ** 3
        grid_vals = torch.randn(grid.total_voxels, 3).to(device).to(dtype)

        for pool_factor in (1, 2, 3, 4, 5, 7, 15, 10):
            our_pooling = fvnn.MaxPool(pool_factor)
            our_pooling_input = fvnn.VDBTensor(grid, grid.jagged_like(grid_vals))
            our_pooling_output = our_pooling(our_pooling_input)
            grid_vals_coarse = our_pooling_output.feature.jdata
            grid_coarse = our_pooling_output.grid
            self.assertTrue(torch.allclose(grid_coarse.voxel_sizes[0], grid.voxel_sizes[0] * pool_factor))
            self.assertTrue(torch.allclose(grid_coarse.origins[0], grid.origins[0] + 0.5 * grid.voxel_sizes[0] * (pool_factor - 1)))

            # Pytorch pooling
            torch_pool_op = torch.nn.MaxPool3d(pool_factor, pool_factor, ceil_mode=True)
            # We compy everything to the CPU because it's noticeably faster to iterate and copy this way
            grid_vals_t = torch.zeros(gsize, gsize, gsize, 3).to(device='cpu', dtype=dtype)
            grid_ijk_cpu = grid.ijk.jdata.cpu()
            grid_vals_cpu = grid_vals.cpu()
            for i, coord in enumerate(grid_ijk_cpu):
                grid_vals_t[coord[0], coord[1], coord[2]] = grid_vals_cpu[i]
            grid_vals_t = grid_vals_t.to(device)
            grid_vals_t = grid_vals_t.permute(3, 0, 1, 2).contiguous()
            grid_vals_t_coarse = torch_pool_op(grid_vals_t.unsqueeze(0)).squeeze()

            grid_vals_coarse_t_flat = torch.zeros_like(grid_vals_coarse, device='cpu')
            grid_coarse_ijk_cpu = grid_coarse.ijk.jdata.cpu()
            for i, coord in enumerate(grid_coarse_ijk_cpu):
                grid_vals_coarse_t_flat[i] = grid_vals_t_coarse[:, coord[0], coord[1], coord[2]]
            grid_vals_coarse_t_flat = grid_vals_coarse_t_flat.to(device)
            self.assertTrue(torch.all(grid_vals_coarse == grid_vals_coarse_t_flat))

    def test_vdbtensor_dense(self):
        dense_tensor = torch.randn(8, 32, 32, 32, 3)
        vdb_tensor = fvnn.VDBTensor.from_dense(dense_tensor)
        self.assertEqual(vdb_tensor.grid_count, 8)
        self.assertEqual(vdb_tensor.total_voxels, 32 ** 3 * 8)
        self.assertTrue(torch.allclose(dense_tensor, vdb_tensor.to_dense()))

    def test_vdbtensor_arithmetic(self):
        grid = fvdb.sparse_grid_from_dense(8, [32, 32, 32], voxel_sizes=0.05, origins=(0.0, 0.0, 0.0))
        v1 = fvnn.VDBTensor(grid, grid.jagged_like(torch.randn(grid.total_voxels, 3)))
        v2 = fvnn.VDBTensor(grid, grid.jagged_like(torch.randn(grid.total_voxels, 3)))

        v = fvnn.cat([v1, v2])
        self.assertTrue(v.grid_count == 16)
        self.assertTrue(torch.allclose(v.feature.jdata, torch.cat([v1.feature.jdata, v2.feature.jdata])))

        v = fvnn.cat([v1, v2], dim=1)
        self.assertTrue(v.grid_count == 8)
        self.assertTrue(torch.allclose(v.feature.jdata, torch.cat([v1.feature.jdata, v2.feature.jdata], dim=1)))

        v = v1 + v2
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata + v2.feature.jdata))

        v = v1 + 1.0
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata + 1.0))

        v = v1 - v2
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata - v2.feature.jdata))

        v = v1 * 2.0
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata * 2.0))

        v = v1 / 2.0
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata / 2.0))

        v = v1 / v2
        self.assertTrue(torch.allclose(v.feature.jdata, v1.feature.jdata / v2.feature.jdata))

    def test_conv_backends(self):
        dtype, device = torch.float32, 'cuda'
        grid = fvdb.sparse_grid_from_points(fvdb.JaggedTensor([
            torch.rand(1024, 3, device=device, dtype=dtype) * 2.0 - 1.0
            for _ in range(8)
        ]), voxel_sizes=[0.025] * 3, origins=[0.0] * 3)
        feature = grid.jagged_like(torch.randn(grid.total_voxels, 16, device=device, dtype=dtype))
        conv_layer = fvnn.SparseConv3d(16, 32).to(device=device, dtype=dtype)

        conv_layer.backend = "default"
        out_feature_default = conv_layer(fvnn.VDBTensor(grid, feature)).feature.jdata

        conv_layer.backend = "halo"
        out_feature_halo = conv_layer(fvnn.VDBTensor(grid, feature)).feature.jdata

        rel_diff_halo = torch.linalg.norm(out_feature_halo - out_feature_default) / \
            torch.linalg.norm(out_feature_default)
        self.assertLess(rel_diff_halo, 1e-3)

        conv_layer.backend = "dense"
        out_feature_dense = conv_layer(fvnn.VDBTensor(grid, feature)).feature.jdata

        rel_diff_dense = torch.linalg.norm(out_feature_dense - out_feature_default) / \
            torch.linalg.norm(out_feature_default)
        self.assertLess(rel_diff_dense, 1e-3)

    @test_expand(list(itertools.product(
        ['cpu', 'cuda'],
        [torch.float32, torch.float64],
        [1, 8],
        [16, 32]
    )))
    def test_simple_net(self, device, dtype, batch_size, resolution):
        torch.random.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        fvnn.SparseConv3d.allow_tf32 = False

        dense_unet = MyUNet(is_fvdb=False).to(device).to(dtype).requires_grad_(True).train()
        dense_in_feature = torch.randn(batch_size, 3, resolution, resolution, resolution, device=device, dtype=dtype)
        dense_out_feature = dense_unet(dense_in_feature)
        dense_out_feature.sum().backward()
        dense_out_grad = {name: param.grad.clone() for name, param in dense_unet.named_parameters()
                          if param.grad is not None}

        sparse_unet = MyUNet(is_fvdb=True).to(device).to(dtype).requires_grad_(True).train()
        copy_weights(sparse_unet, dense_unet)
        sparse_in_feature = fvnn.VDBTensor.from_dense(dense_in_feature.permute(0, 4, 3, 2, 1))

        sparse_out_feature = sparse_unet(sparse_in_feature)
        sparse_out_feature_dense = sparse_out_feature.to_dense().permute(0, 4, 3, 2, 1)
        sparse_out_feature.feature.jdata.sum().backward()
        sparse_out_grad = {name: param.grad.clone() for name, param in sparse_unet.named_parameters()
                           if param.grad is not None}

        rel_error = torch.linalg.norm(sparse_out_feature_dense - dense_out_feature) / \
            torch.linalg.norm(dense_out_feature)
        # print("Max relative difference (sparse vs dense is %.2e)" % rel_error.item())
        self.assertTrue(rel_error < 1.0e-2)

        grad_rel_error: List[float] = []
        for dense_name, dense_param in dense_out_grad.items():
            sparse_param = sparse_out_grad[dense_name]
            abs_error = torch.linalg.norm(dense_param - sparse_param)
            abs_norm = torch.linalg.norm(dense_param)
            rel_error = abs_error / abs_norm
            grad_rel_error.append(rel_error.item())
        # print("Max relative gradient difference (sparse vs dense is %.2e)" % max(grad_rel_error))
        self.assertTrue(max(grad_rel_error) < 1.0e-2)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @parameterized.expand(all_device_dtype_combos)
    def test_conv_loop(self, device, dtype, mutable):
        # NOTE: This test is to test for a problem experienced in issue #287
        #       which was caused by problems with our python bindings + typecasters
        #       the indication that this test is working is that it doesn't segfault
        torch.random.manual_seed(0)
        for _ in range(50):

            pts = torch.empty((10_000, 3), device=device).normal_()
            coords = torch.floor(pts / 0.01).to(torch.int32)

            grid = GridBatch(mutable=mutable, device=device)
            grid.set_from_ijk(coords)

            feature = torch.empty((grid.total_voxels, 4), device=device, dtype=dtype).normal_()

            example_inputs = fvnn.VDBTensor(grid, fvdb.JaggedTensor([feature]))

            model = fvnn.SparseConv3d(in_channels=4, out_channels=32).to(device=device, dtype=dtype)

            model(example_inputs)
if __name__ == '__main__':
    unittest.main()
