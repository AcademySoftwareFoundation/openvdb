import os
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb

all_device_dtype_combos = [
    ['cpu', torch.float32],
    ['cuda', torch.float32],
]

NVOX = 10_000

class TestJaggedTensor(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_like(self, device, dtype):
        num_grids = np.random.randint(1, 128)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)
        featdata = torch.randn(randpts.jdata.shape[0], 32, dtype=torch.float64, device=device)

        randfeats = randpts.jagged_like(featdata)
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.device, randpts.device)
        self.assertEqual(randpts.dtype, torch.float32)
        self.assertEqual(randfeats.dtype, torch.float64)

        featdata = torch.randn(randpts.jdata.shape[0], 32, dtype=torch.float64, device='cuda' if device == 'cpu' else 'cpu')
        randfeats = randpts.jagged_like(featdata)
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.device, randpts.device)
        self.assertEqual(randpts.dtype, torch.float32)
        self.assertEqual(randfeats.dtype, torch.float64)

    @parameterized.expand(all_device_dtype_combos)
    def test_r_masked_select(self, device, dtype):
        num_grids = np.random.randint(1, 128)
        nvox_per_grid = NVOX if device == 'cuda' else 100
        nrand = 10_000 if device == 'cuda' else 100
        pts_list = [torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype) for _ in range(num_grids)]
        randpts = fvdb.JaggedTensor(pts_list)

        mask = torch.rand(randpts.jdata.shape[0], device=device) < 0.5
        masked_randpts = randpts.r_masked_select(mask)
        self.assertEqual(masked_randpts.jdata.shape[0], mask.sum().item())

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_tensor_one_element(self, device, dtype):
        # Make sure we can pass in JaggedTensors with a single thing explicitly
        pts_list = []
        while len(pts_list) == 0:
            pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(4)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device=device, mutable=False)
        gridbatch.set_from_points(randpts, voxel_sizes=0.1)

        grid = gridbatch[0]

        data_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
        ray_o_path = os.path.join(data_path, 'ray_orig.pt')
        ray_d_path = os.path.join(data_path, 'ray_dir.pt')           
        ray_orig = fvdb.JaggedTensor([torch.load(ray_o_path).to(device=device, dtype=dtype)])
        ray_dir = fvdb.JaggedTensor([torch.load(ray_d_path).to(device=device, dtype=dtype)])
        grid.voxels_along_rays(ray_orig, ray_dir, 1)

        ray_orig = torch.load(ray_o_path).to(device=device, dtype=dtype)
        ray_dir = torch.load(ray_d_path).to(device=device, dtype=dtype)
        grid.voxels_along_rays(ray_orig, ray_dir, 1)

    @parameterized.expand(all_device_dtype_combos)
    def test_indexing(self, device, dtype):
        pts_list = []
        while len(pts_list) == 0:
            pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(17)]
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device=device, mutable=False)
        gridbatch.set_from_points(randpts, voxel_sizes=0.1)

        idx = np.random.randint(len(gridbatch))

        self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gridbatch.ijk[idx].jdata))

        self.assertTrue(torch.equal(gridbatch[-4:-2].ijk.jdata, gridbatch.ijk[-4:-2].jdata))

        self.assertTrue(torch.equal(gridbatch[4:-3].ijk.jdata, gridbatch.ijk[4:-3].jdata))

        self.assertTrue(torch.equal(gridbatch[-13:8].ijk.jdata, gridbatch.ijk[-13:8].jdata))

        self.assertTrue(torch.equal(gridbatch[-13:8:1].ijk.jdata, gridbatch.ijk[-13:8:1].jdata))

        self.assertTrue(torch.equal(gridbatch[9:8:1].ijk.jdata, gridbatch.ijk[9:8:1].jdata))

        self.assertTrue(torch.equal(gridbatch.ijk.jdata, gridbatch.ijk[...].jdata))

        self.assertTrue(torch.equal(gridbatch[-900:800].ijk.jdata, gridbatch.ijk[-900:800].jdata))

        self.assertTrue(torch.equal(gridbatch[::].ijk.jdata, gridbatch.ijk[::].jdata))

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[9:8:2])

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[9:8:-1])

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[None])

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[9:8:-1])

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[::-1])

        with self.assertRaises(IndexError):
            print(gridbatch.ijk[::-3])

    @parameterized.expand(all_device_dtype_combos)
    def test_arithmetic_operators(self, device, dtype):
        pts_list = []
        while len(pts_list) == 0:
            pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(17)]
        randpts = fvdb.JaggedTensor(pts_list)
        randpts_b = fvdb.JaggedTensor([torch.rand_like(x) for x in pts_list])

        self.assertTrue(torch.allclose((randpts + randpts_b).jdata, randpts.jdata + randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts + 2).jdata, randpts.jdata + 2))
        self.assertTrue(torch.allclose((randpts + 3.14).jdata, randpts.jdata + 3.14))

        self.assertTrue(torch.allclose((randpts - randpts_b).jdata, randpts.jdata - randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts - 3).jdata, randpts.jdata - 3))

        self.assertTrue(torch.allclose((randpts * randpts_b).jdata, randpts.jdata * randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts * 4).jdata, randpts.jdata * 4))

        self.assertTrue(torch.allclose((randpts / randpts_b).jdata, randpts.jdata / randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts / 5).jdata, randpts.jdata / 5))

        self.assertTrue(torch.allclose((randpts // randpts_b).jdata, randpts.jdata // randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts // 6).jdata, randpts.jdata // 6))

        self.assertTrue(torch.allclose((randpts % randpts_b).jdata, randpts.jdata % randpts_b.jdata))
        self.assertTrue(torch.allclose((randpts % 5).jdata, randpts.jdata % 5))


    @parameterized.expand(all_device_dtype_combos)
    def test_to_devices(self, device, dtype):
        create_device = 'cpu' if device == 'cuda' else 'cuda'
        pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=create_device, dtype=dtype) for _ in range(17)]
        randpts = fvdb.JaggedTensor(pts_list)
        self.assertTrue(randpts.to(device).device, device)

        self.assertTrue(randpts.to(randpts.device).device, randpts.device)

    @parameterized.expand(all_device_dtype_combos)
    def test_batch_size_one(self, device, dtype):
        # Check for issue #181 fixes
        jt = fvdb.JaggedTensor([torch.Tensor([1.0, 2.0, 3.0])])
        jt = jt.jagged_like(jt.jdata)
        self.assertTrue(jt.joffsets.shape, [1,2])

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_ops(self, device, dtype):
        import torch_scatter

        data_list = []
        for b in range(10):
            datum = torch.rand(1000 + np.random.randint(10), 3, 4, device=device, dtype=dtype)
            data_list.append(datum)
        jt = fvdb.JaggedTensor(data_list)

        jt.requires_grad_(True)
        sum_res_ours = jt.jagged_sum()
        grad_out = torch.rand_like(sum_res_ours)
        (sum_res_ours * grad_out).sum().backward()
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        sum_res_ptscatter = torch_scatter.scatter_sum(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.joffsets.shape[0])
        (sum_res_ptscatter * grad_out).sum().backward()
        grad_ptscatter = jt.jdata.grad.clone()

        self.assertTrue(torch.allclose(sum_res_ours, sum_res_ptscatter))
        self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))

        jt.jdata.grad = None
        min_res_ours, _ = jt.jagged_min()
        grad_out = torch.rand_like(min_res_ours)
        (min_res_ours * grad_out).sum().backward()
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        min_res_ptscatter = torch_scatter.scatter_min(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.joffsets.shape[0])[0]
        (min_res_ptscatter * grad_out).sum().backward()
        grad_ptscatter = jt.jdata.grad.clone()

        self.assertTrue(torch.allclose(min_res_ours, min_res_ptscatter))
        self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))

        jt.jdata.grad = None
        max_res_ours, _ = jt.jagged_max()
        grad_out = torch.rand_like(max_res_ours)
        (max_res_ours * grad_out).sum().backward()
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        max_res_ptscatter = torch_scatter.scatter_max(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.joffsets.shape[0])[0]
        (max_res_ptscatter * grad_out).sum().backward()
        grad_ptscatter = jt.jdata.grad.clone()

        self.assertTrue(torch.allclose(max_res_ours, max_res_ptscatter))
        self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))

    def test_sdpa(self):
        torch.random.manual_seed(0)

        # Get dimensions: query (B*Sq, H, D), key (B*Skv, H, D), value (B*Skv, H, T)
        num_heads = 4
        seqlen_q = [10, 20, 30]
        seqlen_kv = [15, 25, 35]
        dim_qk = 64
        dim_v = 128
        scale = 0.5
        device = "cuda"
        dtype = torch.float32

        batch_size = len(seqlen_q)
        q_jagged = fvdb.JaggedTensor([torch.rand(sq, num_heads, dim_qk, device=device, dtype=dtype) for sq in seqlen_q])
        k_jagged = fvdb.JaggedTensor([torch.rand(sk, num_heads, dim_qk, device=device, dtype=dtype) for sk in seqlen_kv])
        v_jagged = fvdb.JaggedTensor([torch.rand(sk, num_heads, dim_v, device=device, dtype=dtype) for sk in seqlen_kv])

        # Torch -- For-loop approach
        out_jagged_torch_forloop = []
        for b in range(batch_size):
            # From LHE to NHLE / SHV to NHSV
            q = q_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)
            k = k_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)
            v = v_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=scale
                )

            # From NHLV to LHV
            out = out.permute(0, 2, 1, 3).squeeze(0)
            out_jagged_torch_forloop.append(out)
        out_jagged_torch_forloop = fvdb.JaggedTensor(out_jagged_torch_forloop)

        # fVDB
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out_jagged_fvdb = fvdb.scaled_dot_product_attention(
                q_jagged, k_jagged, v_jagged, scale=scale
            )

        self.assertTrue(torch.allclose(out_jagged_torch_forloop.jdata, out_jagged_fvdb.jdata))
        self.assertTrue(torch.all(out_jagged_torch_forloop.joffsets == out_jagged_fvdb.joffsets))

    def test_datatype_caster(self):
        feature = torch.randn((120, 32), device='cpu')
        jagged_feature = fvdb.JaggedTensor([feature])

        for i in range(100):
            if jagged_feature.dtype == "torch.float32":
                pass

if __name__ == '__main__':
    unittest.main()
