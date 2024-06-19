import itertools
import unittest

import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torchsparse
import torchsparse.nn.functional as spF
from parameterized import parameterized

from fvdb import JaggedTensor, GridBatch, ConvPackBackend

from .common import random_drop_points_if_mutable, test_expand

all_device_dtype_combos = [
    ['cpu', torch.float32, False, "gather_scatter"],
    ['cuda', torch.float32, False, "gather_scatter"],
    ['cuda', torch.float32, False, "igemm"],
    ['cuda', torch.float32, False, "igemm_sorted"],
    ['cpu', torch.float64, False, "gather_scatter"],
    ['cuda', torch.float64, False, "gather_scatter"],
    ['cpu', torch.float32, True, "gather_scatter"],
    ['cuda', torch.float32, True, "gather_scatter"],
    ['cuda', torch.float32, True, "igemm"],
    ['cuda', torch.float32, True, "igemm_sorted"],
    ['cpu', torch.float64, True, "gather_scatter"],
    ['cuda', torch.float64, True, "gather_scatter"]
]


def build_spconv(grid, kernel_size, stride, backend):
    spconv, target_grid = grid.sparse_conv_kernel_map(kernel_size, stride)
    if backend == "gather_scatter":
        spconv.build_gather_scatter()
        backend_symbol = ConvPackBackend.GATHER_SCATTER
    elif backend == "igemm":
        spconv.build_implicit_gemm(False, 1, True, 1, False)
        backend_symbol = ConvPackBackend.IGEMM
    elif backend == "igemm_sorted":
        spconv.build_implicit_gemm(True, 1, True, 1, False)
        backend_symbol = ConvPackBackend.IGEMM
    else:
        raise ValueError(f"Unknown backend {backend}")
    return spconv, target_grid, backend_symbol


class TestConv(unittest.TestCase):
    @parameterized.expand(all_device_dtype_combos)
    def test_torch_sparse_conv_simple(self, device, dtype, mutable, backend):
        torch.random.manual_seed(0)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, (1, 1, 1))

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        kernel_size = 3
        stride = 1

        vdb_features = torch.ones((int(grid.num_voxels[0].item()), 1), device=device, dtype=dtype)
        vdb_kernels = torch.ones(1, 1, kernel_size, kernel_size, kernel_size,
                                 dtype=dtype, device=device)

        vdb_kernels += torch.randn(1, 1, kernel_size).to(device).to(dtype) * 0.1

        vdb_features.requires_grad = True
        vdb_kernels.requires_grad = True

        spconv, target_grid, symbol = build_spconv(grid, kernel_size, stride, backend)
        out_vdb_features = spconv.sparse_conv_3d(vdb_features, vdb_kernels, symbol)
        out_dense_features = target_grid.read_into_dense(out_vdb_features).squeeze(0).permute(3, 2, 1, 0)

        out_dense_features.sum().backward()

        assert vdb_features.grad is not None  # Fix type errors
        assert vdb_kernels.grad is not None

        vdb_features_grad = torch.clone(vdb_features.grad)
        vdb_kernels_grad = torch.clone(vdb_kernels.grad)

        vdb_features.grad.zero_()
        vdb_kernels.grad.zero_()

        # # Dense convolution & backward
        dense_features = grid.read_into_dense(vdb_features).squeeze(0).permute(3, 2, 1, 0)
        out_dense_features_ref = torch.nn.functional.conv3d(
            dense_features, vdb_kernels, padding=(kernel_size - 1) // 2, stride=stride
        )

        self.assertAlmostEqual(vdb_kernels[0, 0, 1, 1, 1].item(), out_dense_features_ref.squeeze().item())
        self.assertAlmostEqual(vdb_kernels[0, 0, 1, 1, 1].item(), out_dense_features.squeeze().item())
        self.assertAlmostEqual(out_dense_features.squeeze().item(), out_dense_features_ref.squeeze().item())
        out_dense_features_ref.sum().backward()
        dense_features_grad = torch.clone(vdb_features.grad)
        dense_kernels_grad = torch.clone(vdb_kernels.grad)

        self.assertTrue(torch.allclose(out_dense_features, out_dense_features_ref),
                        f"Max dist is {torch.max(out_dense_features - out_dense_features_ref)}")
        self.assertTrue(torch.allclose(vdb_features_grad, dense_features_grad),
                        f"Max dist is {torch.max(vdb_features_grad - dense_features_grad)}")
        self.assertTrue(torch.allclose(vdb_kernels_grad, dense_kernels_grad),
                        f"Max dist is {torch.max(vdb_kernels_grad - dense_kernels_grad)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @test_expand(list(itertools.product(
        ['cuda'],       # torchsparse supports only cuda.
        [torch.float32],
        [1, 4],
        [2, 3, 4, 5, 6],
        [1, 2, 4],
        ["gather_scatter"]
    )))
    def test_torchsparse_conv(self, device, dtype, batch_size, kernel_size, stride, backend):
        torch.random.manual_seed(0)
        in_channel, out_channel = 4, 8

        grid = GridBatch(device=device)
        grid.set_from_points(
            JaggedTensor([torch.randn((100, 3), device=device, dtype=dtype) for _ in range(batch_size)]),
            voxel_sizes=0.05, origins=[0.0] * 3)

        vdb_features = grid.jagged_like(torch.randn((grid.total_voxels, in_channel), device=device, dtype=dtype))
        vdb_kernels = torch.randn(out_channel, in_channel, kernel_size, kernel_size, kernel_size,
                                  dtype=dtype, device=device)

        vdb_features.jdata.requires_grad = True
        vdb_kernels.requires_grad = True

        # Sparse convolution & backward
        spconv, vdb_target_grid, symbol = build_spconv(grid, kernel_size, stride, backend)
        out_vdb_features = spconv.sparse_conv_3d(vdb_features, vdb_kernels, symbol)
        out_vdb_features.jdata.sum().backward()

        assert vdb_features.jdata.grad is not None  # Fix type errors
        assert vdb_kernels.grad is not None

        vdb_features_grad = torch.clone(vdb_features.jdata.grad)
        vdb_kernels_grad = torch.clone(vdb_kernels.grad)

        vdb_features.jdata.grad.zero_()
        vdb_kernels.grad.zero_()

        # torchsparse convolution & backward (kernel arrangement is weird...)
        #   ref: torchsparse@2.0.0b/nn/utils/kernel.py
        ts_tensor = torchsparse.SparseTensor(
            vdb_features.jdata, torch.cat([grid.ijk.jdata, grid.ijk.jidx[:, None]], dim=1))
        if kernel_size % 2 == 0:
            ts_kernel = vdb_kernels.permute(4, 3, 2, 1, 0).reshape(-1, in_channel, out_channel)
        else:
            ts_kernel = vdb_kernels.permute(2, 3, 4, 1, 0).reshape(-1, in_channel, out_channel)
        out_ts_tensor = spF.conv3d(ts_tensor, ts_kernel, kernel_size, stride=stride)
        assert out_ts_tensor.stride[0] == stride

        # Check target grid is aligned.
        ts_target_grid_ijk = JaggedTensor([
            torch.div(out_ts_tensor.coords[out_ts_tensor.coords[:, -1] == b, :3], stride, rounding_mode='floor')
            for b in range(batch_size)])
        idx_map = vdb_target_grid.ijk_to_index(ts_target_grid_ijk)

        # (Optionally: visualize)
        # from pycg import vis
        # vis.show_3d(
        #     [vis.wireframe_bbox(grid.grid_to_world(grid.ijk.jdata - 0.5).jdata,
        #     grid.grid_to_world(grid.ijk.jdata + 0.5).jdata, solid=True, ucid=0)] +
        #     [vis.wireframe_bbox(vdb_target_grid.grid_to_world(vdb_target_grid.ijk.jdata - 0.5).jdata,
        #     vdb_target_grid.grid_to_world(vdb_target_grid.ijk.jdata + 0.5).jdata, solid=True, ucid=1)] +
        #     [vis.wireframe_bbox(vdb_target_grid.grid_to_world(ts_target_grid_ijk.jdata - 0.5).jdata,
        #     vdb_target_grid.grid_to_world(ts_target_grid_ijk.jdata + 0.5).jdata, solid=True, ucid=2)],
        #     use_new_api=True
        # )

        assert idx_map.jdata.shape[0] == vdb_target_grid.total_voxels
        assert torch.all(torch.sort(idx_map.jdata).values == torch.arange(vdb_target_grid.total_voxels, device=device))

        ts_features = out_ts_tensor.feats[vdb_target_grid.ijk_to_inv_index(ts_target_grid_ijk).jdata]
        ts_features.sum().backward()

        dense_features_grad = torch.clone(vdb_features.jdata.grad)
        dense_kernels_grad = torch.clone(vdb_kernels.grad)

        self.assertTrue(torch.allclose(out_vdb_features.jdata, ts_features, atol=1e-6),
                        f"Max dist is {torch.max(out_vdb_features.jdata - ts_features)}")
        self.assertTrue(torch.allclose(vdb_features_grad, dense_features_grad, atol=1e-6),
                        f"Max dist is {torch.max(vdb_features_grad - dense_features_grad)}")
        self.assertTrue(torch.allclose(vdb_kernels_grad, dense_kernels_grad, atol=1e-6),
                        f"Max dist is {torch.max(vdb_kernels_grad - dense_kernels_grad)}")

    @test_expand(list(itertools.product(
        ['cpu', 'cuda'],
        [torch.float32, torch.float64],
        [2, 3, 4, 5, 6],
        [1, 2, 4],
        [False, True],
        ["gather_scatter", "igemm"]
    )))
    def test_torch_sparse_conv(self, device, dtype, kernel_size, stride, mutable, backend):
        torch.random.manual_seed(0)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, (32, 32, 32))
        tol = {}
        tol_grad = {}
        if dtype == torch.float32:
            tol = {'atol': 1e-4}
            tol_grad = {'atol': 1e-3}
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if dtype == torch.float64 or device == "cpu":
            if "igemm" in backend:
                return

        vdb_features = torch.randn((int(grid.num_voxels[0].item()), 4), device=device, dtype=dtype)
        vdb_kernels = torch.randn(8, 4, kernel_size, kernel_size, kernel_size,
                                  dtype=dtype, device=device)

        vdb_features.requires_grad = True
        vdb_kernels.requires_grad = True

        # Dense convolution & backward
        dense_features = grid.read_into_dense(vdb_features).squeeze(0).permute(3, 2, 1, 0)
        out_dense_features_ref = torch.nn.functional.conv3d(
            dense_features, vdb_kernels, padding=(kernel_size - 1) // 2, stride=stride
        )

        out_dense_features_ref.sum().backward()
        assert vdb_features.grad is not None
        assert vdb_kernels.grad is not None
        dense_features_grad = torch.clone(vdb_features.grad)
        dense_kernels_grad = torch.clone(vdb_kernels.grad)

        vdb_features.grad.zero_()
        vdb_kernels.grad.zero_()

        # Sparse convolution & backward
        spconv, target_grid, symbol = build_spconv(grid, kernel_size, stride, backend)
        out_vdb_features = spconv.sparse_conv_3d(vdb_features, vdb_kernels, symbol)
        out_dense_features = target_grid.read_into_dense(out_vdb_features).squeeze(0).permute(3, 2, 1, 0)

        if out_dense_features.size(-1) != out_dense_features_ref.size(-1):
            delta = out_dense_features.size(-1) - out_dense_features_ref.size(-1)
            if stride == 2 and kernel_size == 5:
                out_dense_features = out_dense_features[:, 1:-1, 1:-1, 1:-1]
            else:
                out_dense_features = out_dense_features[:, :-delta, :-delta, :-delta]

        out_dense_features.sum().backward()

        vdb_features_grad = torch.clone(vdb_features.grad)
        vdb_kernels_grad = torch.clone(vdb_kernels.grad)

        # import pdb; pdb.set_trace()
        self.assertTrue(torch.allclose(out_dense_features, out_dense_features_ref, **tol), # type: ignore
                        f"Max dist is {torch.max(out_dense_features - out_dense_features_ref)}")
        self.assertTrue(torch.allclose(vdb_features_grad, dense_features_grad, **tol_grad),  # type: ignore
                        f"Max dist is {torch.max(vdb_features_grad - dense_features_grad)}")
        self.assertTrue(torch.allclose(vdb_kernels_grad, dense_kernels_grad, **tol_grad),  # type: ignore
                        f"Max dist is {torch.max(vdb_kernels_grad - dense_kernels_grad)}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @test_expand(list(itertools.product(
        ['cpu', 'cuda'],
        [torch.float32, torch.float64],
        [ (3,3,3), (1,3,1), (1,3,3), (4,1,4), (5,3,1)],
        [ (1,1,1), (2,2,2), (3,3,3), (1,2,1), (1,3,2)],
        [False ,True],
        ["gather_scatter", "igemm", "igemm_sorted"]
    )))
    def test_torch_sparse_aniso_conv(self, device, dtype, kernel_size:tuple, stride:tuple, mutable, backend):
        torch.random.manual_seed(0)
        grid = GridBatch(mutable=mutable, device=device)
        grid.set_from_dense_grid(1, (32,) * 3)
        tol = {}
        tol_grad = {}
        if dtype == torch.float32:
            tol = {'atol': 1e-4}
            tol_grad = {'atol': 1e-3}
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if dtype == torch.float64 or device == "cpu":
            if "igemm" in backend:
                return

        vdb_features = torch.randn((int(grid.num_voxels[0].item()), 4), device=device, dtype=dtype)
        vdb_kernels = torch.randn(8, 4, kernel_size[0], kernel_size[1], kernel_size[2],
                                  dtype=dtype, device=device)

        vdb_features.requires_grad = True
        vdb_kernels.requires_grad = True

        # Dense convolution & backward
        dense_features = grid.read_into_dense(vdb_features).squeeze(0).permute(3, 2, 1, 0)
        out_dense_features_ref = torch.nn.functional.conv3d(
            dense_features, vdb_kernels, padding=[(k-1)//2 for k in kernel_size],
            stride=stride,
        )

        out_dense_features_ref.sum().backward()
        assert vdb_features.grad is not None
        assert vdb_kernels.grad is not None
        dense_features_grad = torch.clone(vdb_features.grad)
        dense_kernels_grad = torch.clone(vdb_kernels.grad)

        vdb_features.grad.zero_()
        vdb_kernels.grad.zero_()

        # Sparse convolution & backward
        spconv, target_grid, symbol = build_spconv(grid, kernel_size, stride, backend)
        out_vdb_features = spconv.sparse_conv_3d(vdb_features, vdb_kernels, symbol)
        out_dense_features = target_grid.read_into_dense(out_vdb_features).squeeze(0).permute(3, 2, 1, 0)


        for i, (dense_features_dim, ref_features_dim) in enumerate(zip(out_dense_features.shape[-3:], out_dense_features_ref.shape[-3:])):
            if dense_features_dim != ref_features_dim:
                delta = dense_features_dim - ref_features_dim
                if delta > 0:
                    if kernel_size[i]%2==1:
                        l_delta = delta//2 if delta//2>0 else None
                        r_delta = (delta//2)+(delta%2)
                    else:
                        l_delta = None
                        r_delta = delta
                    out_dense_features = out_dense_features[:,
                                                            l_delta if i==0 else None:-r_delta if i==0 else None,
                                                            l_delta if i==1 else None:-r_delta if i==1 else None,
                                                            l_delta if i==2 else None:-r_delta if i==2 else None]
                elif delta < 0:
                    # replace the i'th dimension with -delta/2:delta/2
                    l_delta = (delta//2)+(delta%2)
                    r_delta = delta//2
                    out_dense_features_ref = out_dense_features_ref[:,
                                                            -l_delta if i==0 else None:r_delta if i==0 else None,
                                                            -l_delta if i==1 else None:r_delta if i==1 else None,
                                                            -l_delta if i==2 else None:r_delta if i==2 else None]

        out_dense_features.sum().backward()

        vdb_features_grad = torch.clone(vdb_features.grad)
        vdb_kernels_grad = torch.clone(vdb_kernels.grad)

        # import pdb; pdb.set_trace()
        self.assertTrue(torch.allclose(out_dense_features, out_dense_features_ref, **tol), # type: ignore
                        f"Max dist is {torch.max(out_dense_features - out_dense_features_ref)}")
        self.assertTrue(torch.allclose(vdb_features_grad, dense_features_grad, **tol_grad),  # type: ignore
                        f"Max dist is {torch.max(vdb_features_grad - dense_features_grad)}")
        self.assertTrue(torch.allclose(vdb_kernels_grad, dense_kernels_grad, **tol_grad),  # type: ignore
                        f"Max dist is {torch.max(vdb_kernels_grad - dense_kernels_grad)}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @parameterized.expand(all_device_dtype_combos)
    def test_torch_transposed_sparse_conv(self, device, dtype, mutable, backend):
        torch.random.manual_seed(0)
        source_grid = GridBatch(mutable=mutable, device=device)
        source_grid.set_from_dense_grid(1, (32, 32, 32))
        tol = {}
        tol_grad = {}
        if dtype == torch.float32:
            tol = {'atol': 1e-4}
            tol_grad = {'atol': 1e-3}
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if dtype == torch.float64 or device == "cpu":
            if "igemm" in backend:
                return

        for kernel_size in [2, 3, 4, 5, 6]:
            for stride in [1]:  # FIXME: Test breaks with odd kernel size and even stride
                if kernel_size > 3 and "sorted" in backend:     # Not supported by fast kernel.
                    continue

                kmap, target_grid, symbol = build_spconv(source_grid, kernel_size, stride, backend)
                vdb_features = torch.randn((int(kmap.target_grid.num_voxels[0]), 4), device=device, dtype=dtype)
                vdb_kernels = torch.randn(4, 8, kernel_size, kernel_size, kernel_size,
                                          dtype=dtype, device=device)

                vdb_features.requires_grad = True
                vdb_kernels.requires_grad = True

                # Sparse convolution & backward
                out_vdb_features = kmap.sparse_transpose_conv_3d(vdb_features, vdb_kernels, symbol)
                out_dense_features = source_grid.read_into_dense(out_vdb_features).squeeze(0).permute(3, 2, 1, 0)
                if kernel_size % 2 == 0:
                    pass
                    # Hack to compare with PyTorch with even filter size
                    # out_dense_features = out_dense_features[:, :31, :31, :31]

                out_dense_features.sum().backward()

                assert vdb_features.grad is not None  # Fix type errors
                assert vdb_kernels.grad is not None

                vdb_features_grad = torch.clone(vdb_features.grad)
                vdb_kernels_grad = torch.clone(vdb_kernels.grad)

                vdb_features.grad.zero_()
                vdb_kernels.grad.zero_()

                # Dense convolution & backward
                out_size = 32 + kernel_size - 1 - stride + 1
                out_padding = 32 - out_size if out_size < 32 else 0
                in_padding = (out_size - 32) // 2 if out_size > 32 else 0

                dense_features = target_grid.read_into_dense(vdb_features).squeeze(0).permute(3, 2, 1, 0)
                out_dense_features_ref = torch.nn.functional.conv_transpose3d(
                    dense_features, vdb_kernels, stride=stride, padding=in_padding, output_padding=out_padding
                )

                if out_dense_features_ref.shape[-1] > 32:
                    out_dense_features_ref = out_dense_features_ref[..., :32, :32, :32]

                out_dense_features_ref.sum().backward()
                dense_features_grad = torch.clone(vdb_features.grad)
                dense_kernels_grad = torch.clone(vdb_kernels.grad)

                self.assertTrue(torch.allclose(out_dense_features, out_dense_features_ref, **tol),  # type: ignore
                                f"Max dist is {torch.max(out_dense_features - out_dense_features_ref)}")
                self.assertTrue(torch.allclose(vdb_features_grad, dense_features_grad, **tol_grad),  # type: ignore
                                f"Max dist is {torch.max(vdb_features_grad - dense_features_grad)}")
                self.assertTrue(torch.allclose(vdb_kernels_grad, dense_kernels_grad, **tol_grad),  # type: ignore
                                f"Max dist is {torch.max(vdb_kernels_grad - dense_kernels_grad)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @parameterized.expand(all_device_dtype_combos)
    def test_error_inputs(self, device, dtype, mutable, backend):

        torch.random.manual_seed(0)

        grid = GridBatch(mutable=mutable).to(device)
        grid.set_from_points(torch.randn(1000, 3).to(device).to(dtype), voxel_sizes=0.025, origins=[0.0, 0.0, 0.0])
        random_drop_points_if_mutable(grid)

        def do_conv(feats, kernels):
            kmap, _, symbol = build_spconv(grid, kernels.shape[-1], 1, backend)
            return kmap.sparse_conv_3d(feats, kernels, symbol)

        def do_tranpose_conv(feats, kernels):
            kmap, _, symbol = build_spconv(grid, kernels.shape[-1], 1, backend)
            return kmap.sparse_transpose_conv_3d(feats, kernels, symbol)

        for spconv in [do_conv, do_tranpose_conv]:
            bad_feats = torch.randn(int(grid.num_voxels[0].item()), 0)
            bad_kernels = torch.rand(8, 0, 5, 5, 5, dtype=dtype, device=device)
            with self.assertRaises(ValueError):
                spconv(bad_feats, bad_kernels)


            bad_feats = torch.randn(int(grid.num_voxels[0].item()), 4)
            bad_kernels = torch.rand(8, 0, 5, 5, 5, dtype=dtype, device=device)
            with self.assertRaises(ValueError):
                spconv(bad_feats, bad_kernels)


            bad_feats = torch.randn(int(grid.num_voxels[0].item()), 5)
            bad_kernels = torch.rand(8, 4, 5, 5, 5, dtype=dtype, device=device)
            with self.assertRaises(ValueError):
                spconv(bad_feats, bad_kernels)


            # Bad number of features
            bad_feats = torch.randn(int(grid.num_voxels[0].item()) + 7, 4)
            bad_kernels = torch.rand(8, 4, 5, 5, 5, dtype=dtype, device=device)
            with self.assertRaises(ValueError):
                spconv(bad_feats, bad_kernels)

if __name__ == '__main__':
    unittest.main()
