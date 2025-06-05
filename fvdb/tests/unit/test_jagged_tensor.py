# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import itertools
import tempfile
import unittest
from typing import List

import numpy as np
import torch
import torch_scatter
from parameterized import parameterized

import fvdb
from fvdb.utils.tests import get_fvdb_test_data_path, probabilistic_test

all_device_dtype_combos = [
    ["cuda", torch.float16],
    ["cpu", torch.float32],
    ["cuda", torch.float32],
    ["cpu", torch.float64],
    ["cuda", torch.float64],
]

NVOX = 10_000


class TestJaggedTensor(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(2024)
        np.random.seed(2024)

    def mklol(
        self,
        num_outer,
        num_inner_min,
        num_inner_max,
        device,
        dtype,
        last_dims=(3, 4),
        base_num=1000,
        vary_num=10,
        empty_prob=0.0,
    ):
        pts_list = []
        for _ in range(num_outer):
            pts_list_i = []
            while len(pts_list_i) == 0:
                size = base_num + (np.random.randint(vary_num) if vary_num > 0 else 0)
                if np.random.rand() < empty_prob:
                    size = 0
                pts_list_i = [
                    torch.rand(size, *last_dims, device=device, dtype=dtype)
                    for _ in range(np.random.randint(num_inner_min, num_inner_max))
                ]
            pts_list.append(pts_list_i)
        ret = fvdb.JaggedTensor(pts_list), pts_list
        self.assertTrue(ret[0].eshape == [s for s in ret[0].jdata.shape[1:]])
        return ret

    def mklol_like(self, lol, vary_dim_1=False, vary_dim_2=False):
        res = []
        shape_1 = lol[0][0].shape[1] + np.random.randint(0, 5) if vary_dim_2 else lol[0][0].shape[1]
        for loli in lol:
            res_i = []
            for lolij in loli:
                shape_0 = lolij.shape[0] + np.random.randint(0, 5) if vary_dim_1 else lolij.shape[0]
                res_i.append(torch.rand(shape_0, shape_1, device=lolij.device, dtype=lolij.dtype))
            res.append(res_i)
        return fvdb.JaggedTensor(res), res

    def check_lshape(self, jt: fvdb.JaggedTensor, lt: List[torch.Tensor] | List[List[torch.Tensor]]):
        self.assertEqual(len(jt), len(lt))
        if jt.ldim == 1:
            for i in range(len(jt)):
                self.assertEqual(jt.lshape[i], lt[i].shape[0])
        elif jt.ldim == 2:
            for i, jti in enumerate(jt):
                self.assertEqual(len(jti), len(lt[i]))
                assert isinstance(jt.lshape, list)
                assert isinstance(jti.lshape, list)
                for j in range(len(jti)):
                    assert isinstance(jt.lshape[i], list)
                    self.assertEqual(jt.lshape[i][j], lt[i][j].shape[0])
        else:
            assert False, "jagged tensor ldim should be 1 or 2"

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_noop(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor(tensor_list)

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_noop_list_of_lists(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor([tensor_list, tensor_list])

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_empty_list(self, device, dtype):
        tensor_list = [torch.rand(0, 1, 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor(tensor_list)

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_empty_list_of_lists(self, device, dtype):
        tensor_list = [torch.rand(0, 1, 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor([tensor_list, tensor_list])

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_simple(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor(tensor_list)

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_simple_list_of_lists(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(7)]
        jt = fvdb.JaggedTensor([tensor_list, tensor_list])

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_empty_tensors(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]

        jt = fvdb.JaggedTensor(tensor_list)

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jsqueeze_empty_tensors_list_of_lists(self, device, dtype):
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list += [torch.empty(0, 1, 3, device=device, dtype=dtype) for _ in range(4)]
        tensor_list = [torch.rand(100 + np.random.randint(10), 1, 3, device=device, dtype=dtype) for _ in range(3)]

        jt = fvdb.JaggedTensor([tensor_list, tensor_list])

        jt_squeezed = jt.jsqueeze()

        self.assertEqual(jt_squeezed.lshape, jt.lshape)
        self.assertNotEqual(jt_squeezed.jdata.shape, jt.jdata.shape)
        self.assertTrue(torch.equal(jt_squeezed.jdata.unsqueeze(1), jt.jdata))
        self.assertTrue(torch.equal(jt_squeezed.jdata, jt.jdata.squeeze()))
        self.assertTrue(torch.equal(jt_squeezed.joffsets, jt.joffsets))
        self.assertTrue(torch.equal(jt_squeezed.jidx, jt.jidx))
        self.assertEqual(jt_squeezed.joffsets.shape, jt.joffsets.shape)
        self.assertEqual(jt_squeezed.jidx.shape, jt.jidx.shape)
        self.assertEqual(jt_squeezed.device, jt.device)
        self.assertEqual(jt_squeezed.dtype, jt.dtype)
        self.assertEqual(jt_squeezed.ldim, jt.ldim)
        self.assertNotEqual(jt_squeezed.eshape, jt.eshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jcat_along_dim_0_with_one_tensor(self, device, dtype):
        batch_size = 1

        # Make a point cloud with a random number of points
        def get_pc(num_pc_list: list):
            pc_list = []
            for num_pc in num_pc_list:
                pc_list.append(torch.rand((num_pc, 3)).to(device))
            return pc_list

        num_pc_list = torch.randint(low=50, high=1000, size=(batch_size,), device=device).cpu().tolist()

        pc1_tensor_list = get_pc(num_pc_list)
        pc2_tensor_list = get_pc(num_pc_list)

        pc1_jagged = fvdb.JaggedTensor(pc1_tensor_list)
        pc2_jagged = fvdb.JaggedTensor(pc2_tensor_list)

        cat_dim = 0
        concat_tensor_list = [
            torch.cat([pc1_tensor_list[i], pc2_tensor_list[i]], dim=cat_dim) for i in range(batch_size)
        ]

        jagged_from_concat_list = fvdb.JaggedTensor(concat_tensor_list)
        jcat_result = fvdb.jcat([pc1_jagged, pc2_jagged], dim=cat_dim)

        self.assertTrue(torch.equal(jagged_from_concat_list.jdata, jcat_result.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_pickle(self, device, dtype):
        jt, _ = self.mklol(7, 4, 8, device, dtype)
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(jt, tmp.name)
            jt2: fvdb.JaggedTensor = torch.load(tmp.name)
            self.assertTrue(torch.all(jt.jdata == jt2.jdata))
            self.assertTrue(torch.all(jt.joffsets == jt2.joffsets))
            self.assertTrue(torch.all(jt.jidx == jt2.jidx))
            self.assertTrue(jt.device == jt2.device)
            self.assertTrue(jt.dtype == jt2.dtype)
            self.assertEqual(jt.lshape, jt2.lshape)

        jt = fvdb.JaggedTensor([torch.randn(100 + np.random.randint(10), 3, 2).to(device).to(dtype) for _ in range(10)])
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(jt, tmp.name)
            jt2: fvdb.JaggedTensor = torch.load(tmp.name)
            self.assertTrue(torch.all(jt.jdata == jt2.jdata))
            self.assertTrue(torch.all(jt.joffsets == jt2.joffsets))
            self.assertTrue(torch.all(jt.jidx == jt2.jidx))
            self.assertTrue(jt.device == jt2.device)
            self.assertTrue(jt.dtype == jt2.dtype)
            self.assertEqual(jt.lshape, jt2.lshape)

        jt = fvdb.JaggedTensor([torch.rand(1024, 9, 9, 9)])
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(jt, tmp.name)
            jt2: fvdb.JaggedTensor = torch.load(tmp.name)
            self.assertTrue(torch.all(jt.jdata == jt2.jdata))
            self.assertTrue(torch.all(jt.joffsets == jt2.joffsets))
            self.assertTrue(torch.all(jt.jidx == jt2.jidx))
            self.assertTrue(jt.device == jt2.device)
            self.assertTrue(jt.dtype == jt2.dtype)
            self.assertEqual(jt.lshape, jt2.lshape)

    @parameterized.expand(all_device_dtype_combos)
    def test_jflatten_list_of_lists(self, device, dtype):
        jt1, l1 = self.mklol(7, 4, 8, device, dtype)
        jt2, l2 = self.mklol(3, 7, 11, device, dtype)

        self.check_lshape(jt1, l1)
        self.check_lshape(jt2, l2)

        jt3 = jt1.jflatten(dim=0)
        lshape1 = jt1.lshape
        lshape3 = jt3.lshape
        count = 0
        for i, inner1 in enumerate(jt1):
            for j, inner2 in enumerate(inner1):
                self.assertTrue(torch.all(jt3[count].jdata == inner2.jdata))
                self.assertEqual(lshape1[i][j], lshape3[count])
                count += 1

        jt3 = jt1.jflatten(dim=-2)
        lshape1 = jt1.lshape
        lshape3 = jt3.lshape
        count = 0
        for i, inner1 in enumerate(jt1):
            for j, inner2 in enumerate(inner1):
                self.assertTrue(torch.all(jt3[count].jdata == inner2.jdata))
                self.assertEqual(lshape1[i][j], lshape3[count])
                count += 1

        jt4 = jt2.jflatten(dim=1)
        lshape2 = jt2.lshape
        lshape4 = jt4.lshape
        for i, inner1 in enumerate(jt2):
            data1 = torch.cat(inner1.unbind(), dim=0)
            data2 = jt4[i].jdata
            self.assertTrue(torch.all(data1 == data2))
            self.assertEqual(lshape4[i], np.sum(lshape2[i]))

        jt4 = jt2.jflatten(dim=-1)
        lshape2 = jt2.lshape
        lshape4 = jt4.lshape
        for i, inner1 in enumerate(jt2):
            data1 = torch.cat(inner1.unbind(), dim=0)
            data2 = jt4[i].jdata
            self.assertTrue(torch.all(data1 == data2))
            self.assertEqual(lshape4[i], np.sum(lshape2[i]))

        with self.assertRaises(IndexError):
            jt4 = jt2.jflatten(dim=2)

        with self.assertRaises(IndexError):
            jt4 = jt2.jflatten(dim=-3)

    @parameterized.expand(all_device_dtype_combos)
    def test_jflatten_list(self, device, dtype):
        jt1 = fvdb.jrand([100, 200, 300, 400, 500, 600, 700, 800], [2, 3, 4])

        jt3 = jt1.jflatten(dim=0)
        self.assertEqual(len(jt3.lshape), 1)
        self.assertEqual(jt3.lshape[0], np.sum(jt1.lshape))
        self.assertTrue(torch.all(jt3.unbind()[0] == jt1.jdata))

        jt3 = jt1.jflatten(dim=-1)
        self.assertEqual(len(jt3.lshape), 1)
        self.assertEqual(jt3.lshape[0], np.sum(jt1.lshape))
        self.assertTrue(torch.all(jt3.unbind()[0] == jt1.jdata))

        with self.assertRaises(IndexError):
            jt3 = jt1.jflatten(dim=1)

        with self.assertRaises(IndexError):
            jt3 = jt1.jflatten(dim=-2)

        with self.assertRaises(IndexError):
            jt3 = jt1.jflatten(dim=2)

    @parameterized.expand(all_device_dtype_combos)
    def test_concatenation(self, device, dtype):
        jt1, l1 = self.mklol(
            7,
            2,
            5,
            device,
            dtype,
            last_dims=(3,),
            base_num=1_000_000 if device == "cuda" else 1000,
            vary_num=100,
            empty_prob=0.0,
        )
        jt2, _ = self.mklol(
            3,
            3,
            5,
            device,
            dtype,
            last_dims=(3,),
            base_num=1_000_000 if device == "cuda" else 1000,
            vary_num=100,
            empty_prob=0.0,
        )
        jt3, l3 = self.mklol_like(l1, vary_dim_1=True, vary_dim_2=False)
        jt4, l4 = self.mklol_like(l1, vary_dim_1=False, vary_dim_2=True)

        self.check_lshape(jt1, l1)
        self.check_lshape(jt3, l3)
        self.check_lshape(jt4, l4)

        with self.assertRaises(ValueError):
            jtcat = fvdb.jcat([jt1, jt2], dim=0)

        with self.assertRaises(ValueError):
            jtcat = fvdb.jcat([], dim=0)

        for dim in [-1, 0, 1]:
            jtcat = fvdb.jcat([jt1, jt1], dim=dim)
            lcatted = []
            for i, jtcati in enumerate(jtcat):
                lcatted.append([])
                for j, jtcatij in enumerate(jtcati):
                    cat_ij = torch.cat([l1[i][j], l1[i][j]], dim=dim)  # meow
                    lcatted[-1].append(cat_ij)
                    self.assertTrue(torch.all(jtcatij.jdata == cat_ij))

            jt_to_cat = jt3 if dim == 0 else jt4
            jtcat = fvdb.jcat([jt1, jt1, jt_to_cat, jt1, jt_to_cat, jt1], dim=dim)
            lcatted = []
            for i, jtcati in enumerate(jtcat):
                lcatted.append([])
                for j, jtcatij in enumerate(jtcati):
                    t_test = (l3 if dim == 0 else l4)[i][j]
                    t1ij = l1[i][j]
                    cat_ij = torch.cat([t1ij, t1ij, t_test, t1ij, t_test, t1ij], dim=dim)  # meow
                    lcatted[-1].append(cat_ij)
                    self.assertTrue(torch.all(jtcatij.jdata == cat_ij))

            jtcat = fvdb.jcat([jt1, jt3 if dim == 0 else jt4, jt1], dim=dim)
            lcatted = []
            for i, jtcati in enumerate(jtcat):
                lcatted.append([])
                for j, jtcatij in enumerate(jtcati):
                    cat_ij = torch.cat([l1[i][j], (l3 if dim == 0 else l4)[i][j], l1[i][j]], dim=dim)
                    lcatted[-1].append(cat_ij)
                    self.assertTrue(torch.all(jtcatij.jdata == cat_ij))
            self.check_lshape(jtcat, lcatted)

        jtcat = fvdb.jcat([jt1, jt1], dim=1)
        lcatted = []
        for i, jtcati in enumerate(jtcat):
            lcatted.append([])
            for j, jtcatij in enumerate(jtcati):
                cat_ij = torch.cat([l1[i][j], l1[i][j]], dim=1)
                lcatted[-1].append(cat_ij)
                self.assertTrue(torch.all(jtcatij.jdata == cat_ij))
        jtcat = fvdb.jcat([jt1, jt4, jt1], dim=1)
        lcatted = []
        for i, jtcati in enumerate(jtcat):
            lcatted.append([])
            for j, jtcatij in enumerate(jtcati):
                cat_ij = torch.cat([l1[i][j], l4[i][j], l1[i][j]], dim=1)
                lcatted[-1].append(cat_ij)
                self.assertTrue(torch.all(jtcatij.jdata == cat_ij))

        with self.assertRaises(IndexError):
            jtcat = fvdb.jcat([jt1, jt1], dim=-2)
        with self.assertRaises(IndexError):
            jtcat = fvdb.jcat([jt1, jt1], dim=2)
        with self.assertRaises(IndexError):
            jtcat = fvdb.jcat([jt1, jt1], dim=-3)
        with self.assertRaises(IndexError):
            jtcat = fvdb.jcat([jt1, jt1], dim=3)

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_concatenation(self, device, dtype):
        jt1, list1 = self.mklol(7, 4, 8, device, dtype)
        jt2, list2 = self.mklol(3, 7, 11, device, dtype)

        self.check_lshape(jt1, list1)
        self.check_lshape(jt2, list2)

        jt3 = fvdb.jcat([jt1, jt2], dim=None)
        list3 = list1 + list2
        self.check_lshape(jt3, list3)
        for i, jt3i in enumerate(jt3):
            for j, jt3ij in enumerate(jt3i):
                self.assertTrue(torch.all(jt3ij.jdata == list3[i][j]))

        multi = [self.mklol(np.random.randint(3, 7), 4, 8, device, dtype) for _ in range(10)]
        multi_jt = [a[0] for a in multi]
        multi_list = [a[1] for a in multi]
        ll = []
        for l in multi_list:
            ll += l

        jtl = fvdb.jcat(multi_jt, dim=None)
        self.check_lshape(jtl, ll)

        for i, jtli in enumerate(jtl):
            for j, jtlij in enumerate(jtli):
                self.assertTrue(torch.all(jtlij.jdata == ll[i][j]))

        # Nesting dimension mismatch
        jt4 = fvdb.JaggedTensor([torch.randn(np.random.randint(4, 100), 4, device=device, dtype=dtype)] * 7)
        with self.assertRaises(ValueError):
            _ = fvdb.jcat([jt1, jt4], dim=None)

        # Device dimension mismatch
        other_device = "cpu" if device == "cuda" else "cuda"
        jt4 = jt1.to(other_device)
        with self.assertRaises(ValueError):
            _ = fvdb.jcat([jt1, jt4], dim=None)

        # Dtype dimension mismatch
        other_dtype = torch.float32 if dtype != torch.float32 else torch.float64
        jt4 = jt1.to(other_dtype)
        with self.assertRaises(ValueError):
            _ = fvdb.jcat([jt1, jt4], dim=None)

        # Empty list
        with self.assertRaises(ValueError):
            _ = fvdb.jcat([], dim=None)

    @parameterized.expand(
        [[*l1, *l2] for l1, l2 in itertools.product(all_device_dtype_combos, all_device_dtype_combos)]
    )
    def test_jagged_like(self, from_device, from_dtype, to_device, to_dtype):
        num_grids = np.random.randint(1, 128)
        nvox_per_grid = NVOX if from_device == "cuda" else 100
        nrand = 10_000 if from_device == "cuda" else 100
        pts_list = [
            torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=from_device, dtype=from_dtype)
            for _ in range(num_grids)
        ]
        randpts = fvdb.JaggedTensor(pts_list)
        featdata = torch.randn(randpts.jdata.shape[0], 32, dtype=to_dtype, device=to_device)

        randfeats = randpts.jagged_like(featdata)
        self.check_lshape(randpts, pts_list)
        self.check_lshape(randfeats, pts_list)
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.jdata.shape[0], randpts.jdata.shape[0])
        self.assertEqual(randfeats.device, randpts.device)  # jagged_like ignore device
        self.assertEqual(randpts.dtype, from_dtype)
        self.assertEqual(randfeats.dtype, to_dtype)

    @parameterized.expand(all_device_dtype_combos)
    def test_rmask(self, device, dtype):
        num_grids = np.random.randint(1, 128)
        nvox_per_grid = NVOX if device == "cuda" else 100
        nrand = 10_000 if device == "cuda" else 100
        pts_list = [
            torch.rand(nvox_per_grid + np.random.randint(nrand), 3, device=device, dtype=dtype)
            for _ in range(num_grids)
        ]
        randpts = fvdb.JaggedTensor(pts_list)
        self.check_lshape(randpts, pts_list)

        mask = torch.rand(randpts.jdata.shape[0], device=device) < 0.5
        masked_randpts = randpts.rmask(mask)
        masked_list = []
        for i, pts in enumerate(pts_list):
            maski = mask[randpts.joffsets[i] : randpts.joffsets[i + 1]]
            masked_list.append(pts[maski])
            self.assertTrue(torch.all(masked_randpts[i].jdata == masked_list[-1]))
        self.check_lshape(masked_randpts, masked_list)
        self.assertEqual(masked_randpts.jdata.shape[0], mask.sum().item())

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_tensor_one_element(self, device, dtype):
        # Make sure we can pass in JaggedTensors with a single thing explicitly
        pts_list = []
        while len(pts_list) == 0:
            pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(4)]
        randpts = fvdb.JaggedTensor(pts_list)
        self.check_lshape(randpts, pts_list)
        gridbatch = fvdb.GridBatch(device=device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.1)

        grid = gridbatch[0]

        data_path = get_fvdb_test_data_path()
        ray_o_path = data_path / "jagged_tensor" / "ray_orig.pt"
        ray_d_path = data_path / "jagged_tensor" / "ray_dir.pt"
        ray_o = torch.load(ray_o_path, weights_only=True).to(device=device, dtype=dtype)
        ray_d = torch.load(ray_d_path, weights_only=True).to(device=device, dtype=dtype)
        ray_orig = fvdb.JaggedTensor([ray_o])
        ray_dir = fvdb.JaggedTensor([ray_d])
        self.check_lshape(ray_orig, [ray_o])
        self.check_lshape(ray_dir, [ray_d])
        grid.voxels_along_rays(ray_orig, ray_dir, 1)

    @parameterized.expand(all_device_dtype_combos)
    def test_indexing(self, device, dtype):
        pts_list: List[torch.Tensor] = []
        ijk_list: List[torch.Tensor] = []
        while len(pts_list) == 0:
            for _ in range(17):
                pts = torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) * 10.0
                ijk = fvdb.gridbatch_from_points(pts, voxel_sizes=0.5).ijk.jdata
                ijk_list.append(ijk)
                pts_list.append(pts)
        randpts = fvdb.JaggedTensor(pts_list)
        gridbatch = fvdb.GridBatch(device=device)
        gridbatch.set_from_points(randpts, voxel_sizes=0.5)

        idx = np.random.randint(len(gridbatch))

        self.assertTrue(torch.equal(gridbatch[idx].ijk.jdata, gridbatch.ijk[idx].jdata))
        self.check_lshape(gridbatch[idx].ijk, [ijk_list[idx]])
        self.check_lshape(gridbatch.ijk[idx], [ijk_list[idx]])

        self.assertTrue(torch.equal(gridbatch[-4:-2].ijk.jdata, gridbatch.ijk[-4:-2].jdata))
        self.check_lshape(gridbatch[-4:-2].ijk, ijk_list[-4:-2])
        self.check_lshape(gridbatch.ijk[-4:-2], ijk_list[-4:-2])

        self.assertTrue(torch.equal(gridbatch[4:-3].ijk.jdata, gridbatch.ijk[4:-3].jdata))
        self.check_lshape(gridbatch[4:-3].ijk, ijk_list[4:-3])
        self.check_lshape(gridbatch.ijk[4:-3], ijk_list[4:-3])

        self.assertTrue(torch.equal(gridbatch[-13:8].ijk.jdata, gridbatch.ijk[-13:8].jdata))
        self.check_lshape(gridbatch[-13:8].ijk, ijk_list[-13:8])
        self.check_lshape(gridbatch.ijk[-13:8], ijk_list[-13:8])

        self.assertTrue(torch.equal(gridbatch[-13:8:1].ijk.jdata, gridbatch.ijk[-13:8:1].jdata))
        self.check_lshape(gridbatch[-13:8:1].ijk, ijk_list[-13:8:1])
        self.check_lshape(gridbatch.ijk[-13:8:1], ijk_list[-13:8:1])

        self.assertTrue(torch.equal(gridbatch[9:8:1].ijk.jdata, gridbatch.ijk[9:8:1].jdata))
        # An empty grid returns an ijk JaggedTensor with one thing in it so we can't quite compare!
        # self.check_lshape(gridbatch[9:8:1].ijk, ijk_list[9:8:1])
        self.check_lshape(gridbatch.ijk[9:8:1], ijk_list[9:8:1])

        self.assertTrue(torch.equal(gridbatch[9:8:2].ijk.jdata, gridbatch.ijk[9:8:1].jdata))
        # An empty grid returns an ijk JaggedTensor with one thing in it so we can't quite compare!
        # self.check_lshape(gridbatch[9:8:1].ijk, ijk_list[9:8:1])
        self.check_lshape(gridbatch.ijk[9:8:2], ijk_list[9:8:2])

        self.assertTrue(torch.equal(gridbatch[-13:8:2].ijk.jdata, gridbatch.ijk[-13:8:2].jdata))
        self.check_lshape(gridbatch[-13:8:2].ijk, ijk_list[-13:8:2])
        self.check_lshape(gridbatch.ijk[-13:8:2], ijk_list[-13:8:2])

        self.assertTrue(torch.equal(gridbatch[4:17:3].ijk.jdata, gridbatch.ijk[4:17:3].jdata))
        self.check_lshape(gridbatch[4:17:3].ijk, ijk_list[4:17:3])
        self.check_lshape(gridbatch.ijk[4:17:3], ijk_list[4:17:3])

        self.assertTrue(torch.equal(gridbatch[4:15:4].ijk.jdata, gridbatch.ijk[4:15:4].jdata))
        self.check_lshape(gridbatch[4:15:4].ijk, ijk_list[4:15:4])
        self.check_lshape(gridbatch.ijk[4:15:4], ijk_list[4:15:4])

        self.assertTrue(torch.equal(gridbatch.ijk.jdata, gridbatch.ijk[...].jdata))
        self.check_lshape(gridbatch.ijk, ijk_list)
        self.check_lshape(gridbatch.ijk[...], ijk_list)

        self.assertTrue(torch.equal(gridbatch[-900:800].ijk.jdata, gridbatch.ijk[-900:800].jdata))
        self.check_lshape(gridbatch[-900:800].ijk, ijk_list[-900:800])
        self.check_lshape(gridbatch.ijk[-900:800], ijk_list[-900:800])

        self.assertTrue(torch.equal(gridbatch[::].ijk.jdata, gridbatch.ijk[::].jdata))
        self.check_lshape(gridbatch[::].ijk, ijk_list[::])
        self.check_lshape(gridbatch.ijk[::], ijk_list[::])

        with self.assertRaises(ValueError):
            print(gridbatch.ijk[9:8:0])

        with self.assertRaises(ValueError):
            print(gridbatch.ijk[9:8:-1])

        with self.assertRaises(TypeError):
            print(gridbatch.ijk[None])

        with self.assertRaises(ValueError):
            print(gridbatch.ijk[9:8:-1])

        with self.assertRaises(ValueError):
            print(gridbatch.ijk[::-1])

        with self.assertRaises(ValueError):
            print(gridbatch.ijk[::-3])

    @parameterized.expand(all_device_dtype_combos)
    def test_arithmetic_operators(self, device, dtype):
        pts_list = []
        while len(pts_list) == 0:
            pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype) for _ in range(17)]
        randpts = fvdb.JaggedTensor(pts_list)
        randpts_b = fvdb.JaggedTensor([torch.rand_like(x) + 1e-5 for x in pts_list])

        pts_list_2 = [pts_list[i.item()] for i in torch.randperm(len(pts_list))]
        randpts_c = fvdb.JaggedTensor(pts_list_2)

        self.check_lshape(randpts, pts_list)
        self.check_lshape(randpts_b, pts_list)
        self.check_lshape(randpts_c, pts_list_2)

        # ------------
        # Neg
        # ------------
        res = -randpts
        self.assertTrue(torch.allclose(res.jdata, -randpts.jdata))
        self.check_lshape(res, pts_list)

        # ------------
        # Add
        # ------------
        res = randpts + 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + 2))
        self.check_lshape(res, pts_list)

        res = randpts + 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + 3.14))
        self.check_lshape(res, pts_list)

        res = randpts + randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b + randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata + randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b + randpts_c
        fvdb.config.pedantic_error_checking = False

        # ------------
        # Subtract
        # ------------
        res = randpts - 3
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata - 3))
        self.check_lshape(res, pts_list)

        res = randpts - randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata - randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b - randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata - randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b - randpts_c
        fvdb.config.pedantic_error_checking = False

        # ------------
        # Multiply
        # ------------
        res = randpts * 4
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata * 4))
        self.check_lshape(res, pts_list)

        res = randpts * randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata * randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b * randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata * randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b * randpts_c
        fvdb.config.pedantic_error_checking = False

        # ------------
        # Divide
        # ------------
        res = randpts / randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata / randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b / randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata / randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b / randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts / 5
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata / 5))
        self.check_lshape(res, pts_list)

        # ------------
        # Pow
        # ------------
        res = randpts**randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata**randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b**randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata**randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b**randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts**5
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata**5))
        self.check_lshape(res, pts_list)

        # ------------
        # Floor divide
        # ------------
        res = randpts // randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata // randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b // randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata // randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b // randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts // 6
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata // 6))
        self.check_lshape(res, pts_list)

        # ------------
        # Modulo
        # ------------
        res = randpts % randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata % randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b % randpts_c
        if dtype != torch.float16:  # Not stable in float 16, but not important for this test
            self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata % randpts_c.jdata))
            self.check_lshape(res2, pts_list)
            fvdb.config.pedantic_error_checking = True
            with self.assertRaises(ValueError):
                res = randpts_b % randpts_c
            fvdb.config.pedantic_error_checking = False

        res = randpts % 5
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata % 5))
        self.check_lshape(res, pts_list)

        # ------------
        # Greater than
        # ------------
        res = randpts > randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata > randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b > randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata > randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b > randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts > 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata > 2))
        self.check_lshape(res, pts_list)

        res = randpts > 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata > 3.14))
        self.check_lshape(res, pts_list)

        # ----------------
        # Greater or equal
        # ----------------
        res = randpts >= randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata >= randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b >= randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata >= randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b >= randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts >= 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata >= 2))
        self.check_lshape(res, pts_list)

        res = randpts >= 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata >= 3.14))
        self.check_lshape(res, pts_list)

        # ------------
        # Less than
        # ------------
        res = randpts < randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata < randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b < randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata < randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b < randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts < 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata < 2))
        self.check_lshape(res, pts_list)

        res = randpts < 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata < 3.14))
        self.check_lshape(res, pts_list)

        # ------------------
        # Less than or equal
        # ------------------
        res = randpts <= randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata <= randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b <= randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata <= randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b <= randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts <= 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata <= 2))
        self.check_lshape(res, pts_list)

        res = randpts <= 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata <= 3.14))
        self.check_lshape(res, pts_list)

        # ------------
        # Equals
        # ------------
        res = randpts == randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata == randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b == randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata == randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b == randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts == 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata == 2))
        self.check_lshape(res, pts_list)

        res = randpts == 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata == 3.14))
        self.check_lshape(res, pts_list)

        # ------------
        # Not equals
        # ------------
        res = randpts != randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata != randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts_b != randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts_b.jdata != randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res = randpts_b != randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts != 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata != 2))
        self.check_lshape(res, pts_list)

        res = randpts != 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata != 3.14))
        self.check_lshape(res, pts_list)

    @parameterized.expand(all_device_dtype_combos)
    def test_jreshape(self, device, dtype):
        pts_list_a = []
        for _ in range(4):
            pts_list_i = []
            while len(pts_list_i) == 0:
                pts_list_i = [
                    torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype)
                    for _ in range(np.random.randint(3, 7))
                ]
            pts_list_a.append(pts_list_i)
        pts_list_b = [[torch.rand_like(x) + 1e-5 for x in pts_list_i] for pts_list_i in pts_list_a]
        randpts_a = fvdb.JaggedTensor(pts_list_a)
        self.check_lshape(randpts_a, pts_list_a)

        lshape_a = randpts_a.lshape
        lshape_b = []
        for l in lshape_a:
            lshape_b.extend(l)
        randpts_c = randpts_a.jreshape(lshape_b)
        self.check_lshape(randpts_c, randpts_c.unbind())
        self.assertEqual(randpts_c.lshape, lshape_b)
        self.assertTrue(torch.all(randpts_c.jdata == randpts_a.jdata))
        randpts_a.jdata += 1.0
        self.assertTrue(torch.all(randpts_c.jdata == randpts_a.jdata))

        lshape_a = randpts_a.lshape
        lshape_b = [lshape_a[i.item()] for i in torch.randperm(len(lshape_a))]
        randpts_c = randpts_a.jreshape(lshape_b)
        self.check_lshape(randpts_c, randpts_c.unbind())
        self.assertEqual(randpts_c.lshape, lshape_b)
        self.assertTrue(torch.all(randpts_c.jdata == randpts_a.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_arithmetic_operators_list_of_lists(self, device, dtype):
        pts_list = []
        for _ in range(4):
            pts_list_i = []
            while len(pts_list_i) == 0:
                pts_list_i = [
                    torch.rand(1000 + np.random.randint(10), 3, device=device, dtype=dtype)
                    for _ in range(np.random.randint(3, 7))
                ]
            pts_list.append(pts_list_i)
        pts_list_b = [[torch.rand_like(x) + 1e-5 for x in pts_list_i] for pts_list_i in pts_list]
        randpts = fvdb.JaggedTensor(pts_list)
        randpts_b = fvdb.JaggedTensor(pts_list_b)

        pts_list_c = []
        for l in pts_list:
            pts_list_c.extend(l)
        randpts_c = fvdb.JaggedTensor(pts_list_c)
        self.check_lshape(randpts_c, pts_list_c)
        self.assertTrue(torch.all(randpts_c.joffsets == randpts.joffsets))

        res = -randpts
        self.assertTrue(torch.allclose(res.jdata, -randpts.jdata))
        self.check_lshape(res, pts_list)

        res = randpts + randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts + randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata + randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts + randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts + 2
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + 2))
        self.check_lshape(res, pts_list)

        res = randpts + 3.14
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata + 3.14))
        self.check_lshape(res, pts_list)

        res = randpts - randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata - randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts - randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata - randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts - randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts - 3
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata - 3))
        self.check_lshape(res, pts_list)

        res = randpts * randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata * randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res2 = randpts * randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata * randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts * randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts * 4
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata * 4))
        self.check_lshape(res, pts_list)

        res = randpts / randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata / randpts_b.jdata))
        self.check_lshape(res, pts_list)

        if dtype != torch.float16:  # Not stable in float 16, but not important for this test
            res2 = randpts / randpts_c
            self.assertTrue(torch.allclose(res2.jdata, randpts.jdata / randpts_c.jdata))
            self.check_lshape(res2, pts_list)
            fvdb.config.pedantic_error_checking = True
            with self.assertRaises(ValueError):
                res2 = randpts / randpts_c
            fvdb.config.pedantic_error_checking = False

        res = randpts / 5
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata / 5))
        self.check_lshape(res, pts_list)

        res = randpts // randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata // randpts_b.jdata))
        self.check_lshape(res, pts_list)

        if dtype != torch.float16:  # Not stable in float 16, but not important for this test
            res2 = randpts // randpts_c
            self.assertTrue(torch.allclose(res2.jdata, randpts.jdata // randpts_c.jdata))
            self.check_lshape(res2, pts_list)
            fvdb.config.pedantic_error_checking = True
            with self.assertRaises(ValueError):
                res2 = randpts // randpts_c
            fvdb.config.pedantic_error_checking = False

        res = randpts // 6
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata // 6))
        self.check_lshape(res, pts_list)

        res = randpts % randpts_b
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata % randpts_b.jdata))
        self.check_lshape(res, pts_list)

        res = randpts % 5
        self.assertTrue(torch.allclose(res.jdata, randpts.jdata % 5))
        self.check_lshape(res, pts_list)

        res = randpts > randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata > randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts > randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata > randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts > randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts >= randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata >= randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts >= randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata >= randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts >= randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts < randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata < randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts < randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata < randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts < randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts <= randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata <= randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts <= randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata <= randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts <= randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts == randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata == randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts == randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata == randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts == randpts_c
        fvdb.config.pedantic_error_checking = False

        res = randpts != randpts_b
        self.assertTrue(torch.all(res.jdata == (randpts.jdata != randpts_b.jdata)))
        self.check_lshape(res, pts_list)

        res2 = randpts != randpts_c
        self.assertTrue(torch.allclose(res2.jdata, randpts.jdata != randpts_c.jdata))
        self.check_lshape(res2, pts_list)
        fvdb.config.pedantic_error_checking = True
        with self.assertRaises(ValueError):
            res2 = randpts != randpts_c
        fvdb.config.pedantic_error_checking = False

        for i, rpi in enumerate(randpts):
            for j, _ in enumerate(rpi):
                self.assertTrue(torch.allclose((randpts + randpts_b)[i][j].jdata, pts_list[i][j] + pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts + 2)[i][j].jdata, pts_list[i][j] + 2))
                self.assertTrue(torch.allclose((randpts + 3.14)[i][j].jdata, pts_list[i][j] + 3.14))

                self.assertTrue(torch.allclose((randpts - randpts_b)[i][j].jdata, pts_list[i][j] - pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts - 2)[i][j].jdata, pts_list[i][j] - 2))
                self.assertTrue(torch.allclose((randpts - 3.14)[i][j].jdata, pts_list[i][j] - 3.14))

                self.assertTrue(torch.allclose((randpts * randpts_b)[i][j].jdata, pts_list[i][j] * pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts * 2)[i][j].jdata, pts_list[i][j] * 2))
                self.assertTrue(torch.allclose((randpts * 3.14)[i][j].jdata, pts_list[i][j] * 3.14))

                self.assertTrue(torch.allclose((randpts / randpts_b)[i][j].jdata, pts_list[i][j] / pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts / 2)[i][j].jdata, pts_list[i][j] / 2))
                self.assertTrue(torch.allclose((randpts / 3.14)[i][j].jdata, pts_list[i][j] / 3.14))

                self.assertTrue(torch.allclose((randpts // randpts_b)[i][j].jdata, pts_list[i][j] // pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts // 2)[i][j].jdata, pts_list[i][j] // 2))
                self.assertTrue(torch.allclose((randpts // 3.14)[i][j].jdata, pts_list[i][j] // 3.14))

                self.assertTrue(torch.allclose((randpts % randpts_b)[i][j].jdata, pts_list[i][j] % pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts % 2)[i][j].jdata, pts_list[i][j] % 2))
                self.assertTrue(torch.allclose((randpts % 3.14)[i][j].jdata, pts_list[i][j] % 3.14))

                self.assertTrue(torch.allclose((randpts > randpts_b)[i][j].jdata, pts_list[i][j] > pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts > 2)[i][j].jdata, pts_list[i][j] > 2))
                self.assertTrue(torch.allclose((randpts > 3.14)[i][j].jdata, pts_list[i][j] > 3.14))

                self.assertTrue(torch.allclose((randpts >= randpts_b)[i][j].jdata, pts_list[i][j] >= pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts >= 2)[i][j].jdata, pts_list[i][j] >= 2))
                self.assertTrue(torch.allclose((randpts >= 3.14)[i][j].jdata, pts_list[i][j] >= 3.14))

                self.assertTrue(torch.allclose((randpts < randpts_b)[i][j].jdata, pts_list[i][j] < pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts < 2)[i][j].jdata, pts_list[i][j] < 2))
                self.assertTrue(torch.allclose((randpts < 3.14)[i][j].jdata, pts_list[i][j] < 3.14))

                self.assertTrue(torch.allclose((randpts <= randpts_b)[i][j].jdata, pts_list[i][j] <= pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts <= 2)[i][j].jdata, pts_list[i][j] <= 2))
                self.assertTrue(torch.allclose((randpts <= 3.14)[i][j].jdata, pts_list[i][j] <= 3.14))

                self.assertTrue(torch.allclose((randpts == randpts_b)[i][j].jdata, pts_list[i][j] == pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts == 2)[i][j].jdata, pts_list[i][j] == 2))
                self.assertTrue(torch.allclose((randpts == 3.14)[i][j].jdata, pts_list[i][j] == 3.14))

                self.assertTrue(torch.allclose((randpts != randpts_b)[i][j].jdata, pts_list[i][j] != pts_list_b[i][j]))
                self.assertTrue(torch.allclose((randpts != 2)[i][j].jdata, pts_list[i][j] != 2))
                self.assertTrue(torch.allclose((randpts != 3.14)[i][j].jdata, pts_list[i][j] != 3.14))

    @parameterized.expand(all_device_dtype_combos)
    def test_to_devices(self, device, dtype):
        create_device = "cpu" if device == "cuda" else "cuda"
        pts_list = [torch.rand(1000 + np.random.randint(10), 3, device=create_device, dtype=dtype) for _ in range(17)]
        randpts = fvdb.JaggedTensor(pts_list)
        self.check_lshape(randpts, pts_list)
        self.assertTrue(randpts.to(device).device, device)
        self.check_lshape(randpts.to(device), pts_list)

        self.assertTrue(randpts.to(randpts.device).device, randpts.device)
        self.check_lshape(randpts.to(randpts.device), pts_list)

    @parameterized.expand(all_device_dtype_combos)
    def test_batch_size_one(self, device, dtype):
        # Check for issue #181 fixes
        jlist = [torch.Tensor([1.0, 2.0, 3.0]).to(device=device, dtype=dtype)]
        jt = fvdb.JaggedTensor(jlist)
        self.check_lshape(jt, jlist)
        jt = jt.jagged_like(jt.jdata)
        self.check_lshape(jt, jlist)
        self.assertEqual(jt.joffsets.shape, torch.Size([2]))

    @parameterized.expand(all_device_dtype_combos)
    @probabilistic_test(
        iterations=20,
        pass_percentage=80,
        conditional_args=[
            ["cuda"],
            [torch.float16, torch.float32],
        ],
    )
    def test_jsum(self, device, dtype):
        torch.random.manual_seed(111)
        np.random.seed(111)
        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        for _ in range(10):
            datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
            data_list.append(datum)

        for dim in [-2, -1, 0, 1, 2]:
            keepdim = np.random.rand() > 0.5
            jt = fvdb.JaggedTensor(data_list)
            self.check_lshape(jt, data_list)

            jt.requires_grad_(True)
            sum_res = jt.jsum(dim=dim, keepdim=keepdim)
            sum_res_ours = sum_res.jdata
            sum_list = [l.sum(dim=dim, keepdim=True if dim == 0 else keepdim) for l in data_list]
            self.check_lshape(sum_res, sum_list)

            grad_out = torch.rand_like(sum_res_ours)
            # (sum_res_ours * grad_out).sum().backward()
            sum_res_ours.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ours = jt.jdata.grad.clone()

            jt.jdata.grad = None
            if dim == 0:
                sum_res_ptscatter = torch_scatter.scatter_sum(jt.jdata, jt.jidx.long(), dim=0, dim_size=len(jt))
            else:
                sum_res_ptscatter = jt.jdata.sum(dim=dim, keepdim=keepdim)
            # (sum_res_ptscatter * grad_out).sum().backward()
            sum_res_ptscatter.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ptscatter = jt.jdata.grad.clone()

            tol = {}
            if dtype == torch.float16:
                tol["rtol"] = 1e-1
                tol["atol"] = 1e-1
            elif dtype == torch.float32:
                tol["rtol"] = 1e-4
                tol["atol"] = 1e-5
            _ = torch.where(~torch.isclose(sum_res_ours, sum_res_ptscatter, **tol))
            self.assertTrue(torch.allclose(sum_res_ours, sum_res_ptscatter, **tol))
            self.assertTrue(torch.allclose(grad_ours, grad_ptscatter, **tol))

        with self.assertRaises(IndexError):
            sum_res_ours = jt.jsum(dim=-3)
        with self.assertRaises(IndexError):
            sum_res_ours = jt.jsum(dim=3)
        with self.assertRaises(IndexError):
            sum_res_ours = jt.jsum(dim=-4)
        with self.assertRaises(IndexError):
            sum_res_ours = jt.jsum(dim=4)

    @parameterized.expand(all_device_dtype_combos)
    def test_jmin(self, device, dtype):
        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        for _ in range(10):
            datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
            data_list.append(datum)

        for dim in [-2, -1, 0, 1, 2]:
            keepdim = np.random.rand() > 0.5
            jt = fvdb.JaggedTensor(data_list)
            jt.requires_grad_(True)

            jt.jdata.grad = None
            min_res_ours, _ = jt.jmin(dim=dim, keepdim=keepdim)
            min_list = [l.min(dim=dim, keepdim=True if dim == 0 else keepdim)[0] for l in data_list]
            self.check_lshape(min_res_ours, min_list)
            min_res_ours = min_res_ours.jdata
            grad_out = torch.rand_like(min_res_ours)
            min_res_ours.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ours = jt.jdata.grad.clone()

            jt.jdata.grad = None
            min_res_ptscatter = None

            if dim == 0:
                min_res_ptscatter = torch_scatter.scatter_min(jt.jdata, jt.jidx.long(), dim=0, dim_size=len(jt))[0]
            else:
                min_res_ptscatter = torch.min(jt.jdata, dim=dim, keepdim=keepdim)[0]
            min_res_ptscatter.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ptscatter = jt.jdata.grad.clone()

            self.assertTrue(torch.allclose(min_res_ours, min_res_ptscatter))
            # In the case of multiple identical minima, we may backprop through different values
            if torch.allclose(grad_ours, grad_ptscatter):
                self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))
            else:
                zgours = torch.sort(grad_ours[grad_ours != 0.0])[0]
                zgcmp = torch.sort(grad_ptscatter[grad_ptscatter != 0.0])[0]
                self.assertTrue(torch.allclose(zgours, zgcmp))

        with self.assertRaises(IndexError):
            _ = jt.jmin(dim=-3)
        with self.assertRaises(IndexError):
            _ = jt.jmin(dim=3)
        with self.assertRaises(IndexError):
            _ = jt.jmin(dim=-4)
        with self.assertRaises(IndexError):
            _ = jt.jmin(dim=4)

    @parameterized.expand(all_device_dtype_combos)
    def test_jmax(self, device, dtype):
        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        for _ in range(10):
            datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
            data_list.append(datum)

        for dim in [-2, -1, 0, 1, 2]:
            keepdim = np.random.rand() > 0.5
            jt = fvdb.JaggedTensor(data_list)
            jt.requires_grad_(True)

            jt.jdata.grad = None
            max_res_ours, _ = jt.jmax(dim=dim, keepdim=keepdim)
            max_list = [l.max(dim=dim, keepdim=True if dim == 0 else keepdim)[0] for l in data_list]
            self.check_lshape(max_res_ours, max_list)
            max_res_ours = max_res_ours.jdata

            grad_out = torch.rand_like(max_res_ours)
            max_res_ours.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ours = jt.jdata.grad.clone()

            jt.jdata.grad = None
            if dim == 0:
                max_res_ptscatter = torch_scatter.scatter_max(jt.jdata, jt.jidx.long(), dim=0, dim_size=len(jt))[0]
            else:
                max_res_ptscatter = torch.max(jt.jdata, dim=dim, keepdim=keepdim)[0]
            max_res_ptscatter.backward(grad_out)
            assert jt.jdata.grad is not None
            grad_ptscatter = jt.jdata.grad.clone()

            self.assertTrue(torch.allclose(max_res_ours, max_res_ptscatter))
            if torch.allclose(grad_ours, grad_ptscatter):
                self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))
            else:
                zgours = torch.sort(grad_ours[grad_ours != 0.0])[0]
                zgcmp = torch.sort(grad_ptscatter[grad_ptscatter != 0.0])[0]
                self.assertTrue(torch.allclose(zgours, zgcmp))
        with self.assertRaises(IndexError):
            _ = jt.jmax(dim=-3)
        with self.assertRaises(IndexError):
            _ = jt.jmax(dim=3)
        with self.assertRaises(IndexError):
            _ = jt.jmax(dim=-4)
        with self.assertRaises(IndexError):
            _ = jt.jmax(dim=4)

    @parameterized.expand(all_device_dtype_combos)
    def test_jmin_list_of_lists(self, device, dtype):
        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        count = 0
        for b in range(10):
            data_list.append([])
            num_inner_lists_i = np.random.randint(3, 7)
            for _ in range(num_inner_lists_i):
                datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
                data_list[-1].append(datum)
                count += 1
        jt = fvdb.JaggedTensor(data_list)
        jt.requires_grad_(True)

        jt.jdata.grad = None
        min_val_ours, min_idx_ours = jt.jmin()
        min_list = [[l.min().unsqueeze(0) for l in dl] for dl in data_list]
        self.check_lshape(min_val_ours, min_list)
        index_mismatch = False
        catted_min_idx = []
        for i, mvoi in enumerate(min_val_ours):
            for j, mvoij in enumerate(mvoi):
                min_res_i, min_idx_i = data_list[i][j].min(0)
                self.assertTrue(torch.all(mvoij.jdata.squeeze() == min_res_i.squeeze()))
                self.assertTrue(torch.all(min_idx_ours.jdata >= 0), str(min_idx_ours.jdata.min()))
                catted_min_idx.append(min_idx_i)
                if not torch.all(min_idx_ours[i][j].jdata == min_idx_i).item():
                    index_mismatch = True
                    assert len(min_idx_i.shape) == 2
                    # This is a bit of kludge but sometimes the indices mismatch if there are multiple maximum values.
                    # So we check that when this happens that the indices are still validly referring to the minimum
                    # values in the tensor.
                    for a in range(min_idx_i.shape[0]):
                        for b in range(min_idx_i.shape[1]):
                            if min_idx_ours[i][j].jdata[0][a][b] != min_idx_i[a][b]:
                                idx_ours = min_idx_ours[i][j].jdata[0][a][b].item()
                                idx_i = min_idx_i[a][b].item()
                                self.assertEqual(jt[i][j].jdata[idx_i, a, b], data_list[i][j][idx_i, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_ours, a, b], data_list[i][j][idx_ours, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_i, a, b], data_list[i][j][idx_ours, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_ours, a, b], data_list[i][j][idx_i, a, b])
                else:
                    self.assertTrue(
                        torch.all(min_idx_ours[i][j].jdata == min_idx_i).item(),
                        str(min_idx_ours[i][j].jdata) + " vs " + str(min_idx_i),
                    )
        catted_min_idx = torch.stack(catted_min_idx, dim=0)

        min_res_jdata = min_val_ours.jdata
        grad_out = torch.rand_like(min_res_jdata)
        min_res_jdata.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        min_res_ptscatter = torch_scatter.scatter_min(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.num_tensors)[0]
        min_res_ptscatter.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ptscatter = jt.jdata.grad.clone()

        self.assertTrue(torch.allclose(min_res_jdata, min_res_ptscatter))
        if not index_mismatch:
            zgours = torch.sort(grad_ours[grad_ours != 0.0])[0]
            zgcmp = torch.sort(grad_ptscatter[grad_ptscatter != 0.0])[0]
            self.assertTrue(torch.allclose(zgours, zgcmp))
            self.assertTrue(
                torch.allclose(grad_ours, grad_ptscatter),
                str((grad_ours[grad_ours != 0] - grad_ptscatter[grad_ptscatter != 0]).max()),
            )
        else:
            zgours = torch.sort(grad_ours[grad_ours != 0.0])[0]
            zgcmp = torch.sort(grad_ptscatter[grad_ptscatter != 0.0])[0]
            self.assertTrue(torch.allclose(zgours, zgcmp))

    @parameterized.expand(all_device_dtype_combos)
    def test_jmax_list_of_lists(self, device, dtype):
        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        count = 0
        for b in range(10):
            data_list.append([])
            num_inner_lists_i = np.random.randint(3, 7)
            for _ in range(num_inner_lists_i):
                datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
                data_list[-1].append(datum)
                count += 1
        jt = fvdb.JaggedTensor(data_list)
        jt.requires_grad_(True)
        jt.jdata.grad = None
        max_val_ours, max_idx_ours = jt.jmax()
        max_list = [[l.max().unsqueeze(0) for l in dl] for dl in data_list]
        self.check_lshape(max_val_ours, max_list)
        index_mismatch = False
        for i, mvoi in enumerate(max_val_ours):
            for j, mvoij in enumerate(mvoi):
                max_res_i, max_idx_i = data_list[i][j].max(0)
                self.assertTrue(torch.allclose(mvoij.jdata, max_res_i))
                if not torch.all(max_idx_ours[i][j].jdata == max_idx_i).item():
                    index_mismatch = True
                    assert len(max_idx_i.shape) == 2
                    # This is a bit of kludge but sometimes the indices mismatch if there are multiple maximum values.
                    # So we check that when this happens that the indices are still validly referring to the maximum
                    # values in the tensor.
                    for a in range(max_idx_i.shape[0]):
                        for b in range(max_idx_i.shape[1]):
                            if max_idx_ours[i][j].jdata[0][a][b] != max_idx_i[a][b]:
                                idx_ours = max_idx_ours[i][j].jdata[0][a][b].item()
                                idx_i = max_idx_i[a][b].item()
                                self.assertEqual(jt[i][j].jdata[idx_i, a, b], data_list[i][j][idx_i, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_ours, a, b], data_list[i][j][idx_ours, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_i, a, b], data_list[i][j][idx_ours, a, b])
                                self.assertEqual(jt[i][j].jdata[idx_ours, a, b], data_list[i][j][idx_i, a, b])
                else:
                    self.assertTrue(
                        torch.all(max_idx_ours[i][j].jdata == max_idx_i).item(),
                        str(max_idx_ours[i][j].jdata) + " vs " + str(max_idx_i),
                    )

        max_res_jdata = max_val_ours.jdata
        grad_out = torch.rand_like(max_res_jdata)
        max_res_jdata.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        max_res_ptscatter = torch_scatter.scatter_max(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.num_tensors)[0]
        max_res_ptscatter.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ptscatter = jt.jdata.grad.clone()

        self.assertTrue(torch.allclose(max_res_jdata, max_res_ptscatter))
        if not index_mismatch:
            self.assertTrue(torch.allclose(grad_ours, grad_ptscatter))
        else:
            zgours = torch.sort(grad_ours[grad_ours != 0.0])[0]
            zgcmp = torch.sort(grad_ptscatter[grad_ptscatter != 0.0])[0]
            self.assertTrue(torch.allclose(zgours, zgcmp))

    @parameterized.expand(all_device_dtype_combos)
    def test_jsum_list_of_lists(self, device, dtype):
        tol = {}
        if dtype == torch.float16:
            tol["rtol"] = 1e-1
            tol["atol"] = 1e-1
        elif dtype == torch.float32:
            tol["rtol"] = 1e-3
            tol["atol"] = 1e-4

        if dtype == torch.float16:
            min_num = 100
        else:
            min_num = 1000
        data_list = []
        count = 0
        for _ in range(10):
            data_list.append([])
            num_inner_lists_i = np.random.randint(3, 7)
            for _ in range(num_inner_lists_i):
                datum = torch.randn(min_num + np.random.randint(10), 3, 4, device=device, dtype=dtype)
                data_list[-1].append(datum)
                count += 1
        jt = fvdb.JaggedTensor(data_list)
        jt.requires_grad_(True)

        sum_ours = jt.jsum()
        sum_list = [[l.sum().unsqueeze(0) for l in dl] for dl in data_list]
        self.check_lshape(sum_ours, sum_list)
        for i, soi in enumerate(sum_ours):
            for j, soij in enumerate(soi):
                self.assertTrue(
                    torch.allclose(soij.jdata, data_list[i][j].sum(0), **tol),
                    str(soij.jdata) + " vs " + str(data_list[i][j].sum(0)),
                )

        sum_ours_jdata = jt.jsum().jdata
        grad_out = torch.rand_like(sum_ours_jdata)
        # (sum_ours_jdata * grad_out).sum().backward()
        sum_ours_jdata.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ours = jt.jdata.grad.clone()

        jt.jdata.grad = None
        sum_res_ptscatter = torch_scatter.scatter_sum(jt.jdata, jt.jidx.long(), dim=0, dim_size=jt.num_tensors)
        # (sum_res_ptscatter * grad_out).sum().backward()
        sum_res_ptscatter.backward(grad_out)
        assert jt.jdata.grad is not None
        grad_ptscatter = jt.jdata.grad.clone()

        _ = torch.where(~torch.isclose(sum_ours_jdata, sum_res_ptscatter, **tol))
        self.assertTrue(torch.allclose(sum_ours_jdata, sum_res_ptscatter, **tol))
        self.assertTrue(torch.allclose(grad_ours, grad_ptscatter, **tol))

    @parameterized.expand([(torch.float16,), (torch.float32,), (torch.float64,)])
    def test_sdpa(self, dtype):
        torch.random.manual_seed(0)

        # Get dimensions: query (B*Sq, H, D), key (B*Skv, H, D), value (B*Skv, H, T)
        num_heads = 4
        seqlen_q = [10, 20, 30]
        seqlen_kv = [15, 25, 35]
        dim_qk = 64
        dim_v = 128
        scale = 0.5
        device = "cuda"

        batch_size = len(seqlen_q)
        q_list = [torch.rand(sq, num_heads, dim_qk, device=device, dtype=dtype) for sq in seqlen_q]
        k_list = [torch.rand(sk, num_heads, dim_qk, device=device, dtype=dtype) for sk in seqlen_kv]
        v_list = [torch.rand(sk, num_heads, dim_v, device=device, dtype=dtype) for sk in seqlen_kv]
        q_jagged = fvdb.JaggedTensor(q_list)
        k_jagged = fvdb.JaggedTensor(k_list)
        v_jagged = fvdb.JaggedTensor(v_list)
        self.check_lshape(q_jagged, q_list)
        self.check_lshape(k_jagged, k_list)
        self.check_lshape(v_jagged, v_list)

        # Torch -- For-loop approach
        out_jagged_torch_forloop_list = []
        for b in range(batch_size):
            # From LHE to NHLE / SHV to NHSV
            q = q_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)
            k = k_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)
            v = v_jagged[b].jdata.unsqueeze(0).permute(0, 2, 1, 3)

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

            # From NHLV to LHV
            out = out.permute(0, 2, 1, 3).squeeze(0)
            out_jagged_torch_forloop_list.append(out)
        out_jagged_torch_forloop = fvdb.JaggedTensor(out_jagged_torch_forloop_list)
        self.check_lshape(out_jagged_torch_forloop, out_jagged_torch_forloop_list)

        # fVDB
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out_jagged_fvdb = fvdb.scaled_dot_product_attention(q_jagged, k_jagged, v_jagged, scale=scale)
            self.check_lshape(out_jagged_fvdb, out_jagged_torch_forloop_list)
        self.assertTrue(torch.allclose(out_jagged_torch_forloop.jdata, out_jagged_fvdb.jdata))
        self.assertTrue(torch.all(out_jagged_torch_forloop.joffsets == out_jagged_fvdb.joffsets))

    def test_datatype_caster(self):
        feature = torch.randn((120, 32), device="cpu")
        jagged_feature = fvdb.JaggedTensor([feature])
        self.check_lshape(jagged_feature, [feature])

        for _ in range(100):
            if jagged_feature.dtype == "torch.float32":
                pass

    def test_unbind(self):
        lt = [torch.randn(np.random.randint(100, 200), 7) for _ in range(11)]
        jt = fvdb.JaggedTensor(lt)
        self.check_lshape(jt, lt)

        lt2 = jt.unbind()
        self.check_lshape(jt, lt2)
        self.assertEqual(len(lt), len(lt2))
        for i, lti in enumerate(lt):
            self.assertTrue(torch.all(lti == lt2[i]).item())

        lt = [[torch.randn(np.random.randint(100, 200), 7) for _ in range(11)] for _ in range(11)]
        jt = fvdb.JaggedTensor(lt)
        self.check_lshape(jt, lt)
        lt2 = jt.unbind()
        self.check_lshape(jt, lt2)
        self.assertEqual(len(lt), len(lt2))
        for i, lti in enumerate(lt):
            self.assertEqual(len(lti), len(lt2[i]))
            for j, ltij in enumerate(lti):
                self.assertTrue(torch.all(ltij == lt2[i][j]).item())

    def test_list_of_lists_indexing(self):
        lt = [
            [torch.randn(np.random.randint(100, 200), 7) for _ in range(int(l.item()))]
            for l in torch.randint(3, 17, (7,))
        ]
        jt = fvdb.JaggedTensor(lt)
        self.check_lshape(jt, lt)
        lt2 = jt.unbind()
        self.check_lshape(jt, lt2)
        self.assertEqual(len(lt), len(lt2))
        for i, li in enumerate(lt):
            self.assertEqual(len(li), len(lt2[i]))
            for j, lij in enumerate(li):
                self.assertTrue(torch.all(lij == lt2[i][j]).item())
                self.assertTrue(torch.all(jt[i][j].jdata == lt2[i][j]).item())

    @parameterized.expand(["cuda", "cpu"])
    def test_list_of_lists_slicing(self, device):
        lt = [
            [torch.randn(np.random.randint(100, 200), 7).to(device) for _ in range(int(l.item()))]
            for l in torch.randint(3, 5, (10,))
        ]
        jt = fvdb.JaggedTensor(lt)
        self.check_lshape(jt, lt)

        def check_eq(jt_, lt_):
            for i, li in enumerate(lt_):
                self.assertEqual(len(li), len(jt_[i].unbind()))
                for j, lij in enumerate(li):
                    self.assertTrue(torch.all(lij == jt_[i][j].jdata).item())

        lt2 = jt.unbind()
        self.check_lshape(jt, lt2)
        self.assertEqual(len(lt), len(lt2))
        for i, li in enumerate(lt):
            self.assertEqual(len(li), len(lt2[i]))
            for j, lij in enumerate(li):
                self.assertTrue(torch.all(lij == lt2[i][j]).item())
                self.assertTrue(torch.all(jt[i][j].jdata == lt2[i][j]).item())

        jt2 = jt[2:3]
        lt2 = lt[2:3]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:4]
        lt2 = lt[:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:]
        lt2 = lt[:]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:-1]
        lt2 = lt[:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5:]
        lt2 = lt[-5:]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-1]
        lt2 = lt[-1]
        self.check_lshape(jt2, lt2)
        self.assertEqual(jt2.ldim, 1)
        for i, li in enumerate(lt2):
            self.assertEqual(len(li), len(jt2[i].unbind()[0]))
            self.assertTrue(torch.all(li == jt2[i].jdata).item())

        jt2 = jt[1:1]
        lt2 = lt[1:1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-1:1]
        lt2 = lt[-1:1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5:-1]
        lt2 = lt[-5:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5000:-1]
        lt2 = lt[-5000:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5000:5000]
        lt2 = lt[-5000:5000]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[2:8:2]
        lt2 = lt[2:8:2]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:-1:3]
        lt2 = lt[3:-1:3]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:11:4]
        lt2 = lt[3:11:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:2:4]
        lt2 = lt[3:2:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

    @parameterized.expand(["cuda", "cpu"])
    def test_slicing_list_of_lists_small(self, device):
        lt = [
            [torch.randn(0, 7, device=device), torch.randn(2, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(1, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(1, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(1, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(0, 7, device=device)],
            [torch.randn(0, 7, device=device), torch.randn(1, 7, device=device), torch.randn(0, 7, device=device)],
        ]
        jt = fvdb.JaggedTensor(lt)

        def check_eq(jt_, lt_):
            for i, li in enumerate(lt_):
                self.assertEqual(len(li), len(jt_[i].unbind()))
                for j, lij in enumerate(li):
                    self.assertTrue(torch.all(lij == jt_[i][j].jdata).item())

        jt2 = jt[2:3]
        lt2 = lt[2:3]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[2:4]
        lt2 = lt[2:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:4]
        lt2 = lt[:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:]
        lt2 = lt[:]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[:-1]
        lt2 = lt[:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5:]
        lt2 = lt[-5:]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[1:1]
        lt2 = lt[1:1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-1:1]
        lt2 = lt[-1:1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5:-1]
        lt2 = lt[-5:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5000:-1]
        lt2 = lt[-5000:-1]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[-5000:5000]
        lt2 = lt[-5000:5000]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[2:8:2]
        lt2 = lt[2:8:2]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:-1:3]
        lt2 = lt[3:-1:3]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:11:4]
        lt2 = lt[3:11:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

        jt2 = jt[3:2:4]
        lt2 = lt[3:2:4]
        self.check_lshape(jt2, lt2)
        check_eq(jt2, lt2)

    @parameterized.expand(["cuda", "cpu"])
    def test_jagged_tensor_jagged_tensor_indexing_single_tensor_list(self, device):
        t1 = torch.randn(100, 3, device=device)
        l1 = [t1]
        jt1 = fvdb.JaggedTensor(l1)
        pmt1 = torch.randperm(100, device=device)
        lpmt1 = [pmt1]
        jpmt1 = fvdb.JaggedTensor(lpmt1)
        jt_permuted = jt1[jpmt1]
        self.assertTrue(torch.all(jt_permuted.jdata == t1[pmt1]).item())

        t1 = torch.randn(100, 3, device=device)
        l1 = [[t1]]
        jt1 = fvdb.JaggedTensor(l1)
        pmt1 = torch.randperm(100, device=device)
        lpmt1 = [[pmt1]]
        jpmt1 = fvdb.JaggedTensor(lpmt1)
        jt_permuted = jt1[jpmt1]
        self.assertTrue(torch.all(jt_permuted.jdata == t1[pmt1]).item())

        t1 = torch.randn(10, 3, device=device)
        l1 = [torch.zeros(0, 3, device=device), t1, torch.zeros(0, 3, device=device)]
        jt1 = fvdb.JaggedTensor(l1)
        pmt1 = torch.randperm(10, device=device)
        empty_idx = torch.zeros(0, dtype=pmt1.dtype, device=device)
        lpmt1 = [empty_idx, pmt1, empty_idx]
        jpmt1 = fvdb.JaggedTensor(lpmt1)
        jt_permuted = jt1[jpmt1]
        self.assertEqual(jt_permuted.lshape, [0, 10, 0])
        self.assertTrue(torch.all(jt_permuted.jdata == t1[pmt1]).item())

        t1 = torch.randn(10, 3, device=device)
        empty_data = torch.zeros(0, 3, device=device)
        l1 = [
            [empty_data, t1, empty_data],
            [t1, empty_data],
            [empty_data],
            [empty_data, empty_data, t1, empty_data],
            [t1],
        ]
        jt1 = fvdb.JaggedTensor(l1)
        pmt1 = torch.randperm(10, device=device)
        empty_idx = torch.zeros(0, dtype=pmt1.dtype, device=device)
        lpmt1 = [
            [empty_idx, pmt1, empty_idx],
            [pmt1, empty_idx],
            [empty_idx],
            [empty_idx, empty_idx, pmt1, empty_idx],
            [pmt1],
        ]
        jpmt1 = fvdb.JaggedTensor(lpmt1)
        jt_permuted = jt1[jpmt1]
        self.assertEqual(jt_permuted.lshape, [[0, 10, 0], [10, 0], [0], [0, 0, 10, 0], [10]])
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[jpmt1[i][j].jdata] == jtpij.jdata).item())

    @parameterized.expand(["cuda", "cpu"])
    def test_jagged_tensor_integer_indexing(self, device):
        jt1, l1 = self.mklol(7, 4, 8, device, torch.float32)
        self.check_lshape(jt1, l1)

        # Randomly permute the data
        permlist = []
        for jti in jt1:
            pli = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)
                pli.append(pmtij)
            permlist.append(pli)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, l1)
        self.check_lshape(jt_permuted, permlist)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Subsample the data
        permlist = []
        for jti in jt1:
            pli = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)[: np.random.randint(1, 5)]
                pli.append(pmtij)
            permlist.append(pli)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permlist)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Extend
        permlist = []
        for jti in jt1:
            pli = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)[: jtij.rshape[0] // 2]
                pmtij = torch.cat([pmtij, pmtij, pmtij, pmtij])
                pli.append(pmtij)
            permlist.append(pli)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permlist)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Errors:
        # Wrong type
        with self.assertRaises(IndexError):
            _ = jt1[torch.randint(0, 10, (11,)).to(device)]
        # Wrong dtype
        idx = self.mklol(5, 2, 3, device, torch.float32)[0]
        with self.assertRaises(IndexError):
            _ = jt1[idx]
        # Wrong list shape
        randintegers = []
        for li in idx:
            randintegers.append([])
            for _ in li:
                randintegers[-1].append(torch.randint(0, 10, size=(10,)).to(device))
        randintegers = fvdb.JaggedTensor(randintegers)
        with self.assertRaises(IndexError):
            _ = jt1[randintegers]

    @parameterized.expand(["cuda", "cpu"])
    def test_jagged_tensor_integer_indexing_multidim(self, device):
        jt1, _ = self.mklol(7, 4, 8, device, torch.float32)

        # Randomly permute the data
        permlist = []
        permdata = []
        for jti in jt1:
            pli = []
            pdi = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)
                pmtij = torch.stack([pmtij, pmtij], dim=-1)
                pli.append(pmtij)
                pdi.append(jtij.jdata[pmtij])
            permlist.append(pli)
            permdata.append(pdi)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permdata)
        self.check_lshape(permjt, permlist)
        self.check_lshape(jt1, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Subsample the data
        permlist = []
        permdata = []
        for jti in jt1:
            pli = []
            pdi = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)[: np.random.randint(1, 5)]
                pmtij = torch.stack([pmtij, pmtij], dim=-1)
                pli.append(pmtij)
                pdi.append(jtij.jdata[pmtij])
            permlist.append(pli)
            permdata.append(pdi)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permdata)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Extend
        permlist = []
        permdata = []
        for jti in jt1:
            pli = []
            pdi = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)[: jtij.rshape[0] // 2]
                pmtij = torch.cat([pmtij, pmtij, pmtij, pmtij])
                pmtij = torch.stack([pmtij, pmtij], dim=-1)
                pli.append(pmtij)
                pdi.append(jtij.jdata[pmtij])
            permlist.append(pli)
            permdata.append(pdi)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permdata)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

        # Many dimensions
        permlist = []
        permdata = []
        for jti in jt1:
            pli = []
            pdi = []
            for jtij in jti:
                pmtij = torch.randperm(jtij.rshape[0]).to(device)  # [N]
                pmtij = torch.stack([pmtij, pmtij], dim=-1)  # [N, 2]
                pmtij = pmtij.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 4)  # [N, 2, 3, 4]
                pli.append(pmtij)
                pdi.append(jtij.jdata[pmtij])
            permlist.append(pli)
            permdata.append(pdi)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permdata)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())

    @parameterized.expand(["cuda", "cpu"])
    def test_jagged_tensor_boolean_indexing(self, device):
        jt1, l1 = self.mklol(7, 4, 8, device, torch.float32)
        self.check_lshape(jt1, l1)

        # Randomly permute the data
        permlist = []
        permdata = []
        for jti in jt1:
            pli = []
            pdi = []
            for jtij in jti:
                pmtij = torch.randn(jtij.rshape[0]).to(device) > 0.5
                pli.append(pmtij)
                pdi.append(jtij.jdata[pmtij])
            permlist.append(pli)
            permdata.append(pdi)
        permjt = fvdb.JaggedTensor(permlist)
        jt_permuted = jt1[permjt]
        self.check_lshape(jt_permuted, permdata)
        self.check_lshape(permjt, permlist)
        for i, jtpi in enumerate(jt_permuted):
            for j, jtpij in enumerate(jtpi):
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == jtpij.jdata).item())
                self.assertTrue(torch.all(jt1[i][j].jdata[permjt[i][j].jdata] == permdata[i][j]).item())

    def test_edim(self):
        jt, lt = self.mklol(7, 4, 8, "cuda", torch.float32)
        self.check_lshape(jt, lt)
        self.assertEqual(jt.edim, 2)
        self.assertEqual(jt.eshape[0], 3)
        self.assertEqual(jt.eshape[1], 4)

        jt2 = jt.jagged_like(torch.zeros(jt.jdata.shape[0], device=jt.device))
        self.assertEqual(jt2.edim, 0)
        self.assertEqual(len(jt2.eshape), 0)

        jt = fvdb.JaggedTensor(torch.randn(100))
        self.assertEqual(jt.edim, 0)
        self.assertEqual(len(jt.eshape), 0)

    @parameterized.expand(all_device_dtype_combos)
    def test_empty_tensors_and_scalars(self, device, dtype):
        tscalar = torch.tensor([54]).to(device).to(dtype)
        jt1 = fvdb.JaggedTensor([tscalar])
        self.check_lshape(jt1, [tscalar])
        self.assertEqual([s for s in jt1.rshape], [s for s in tscalar.shape])
        self.assertEqual(jt1.eshape, [])

        jt2 = fvdb.JaggedTensor([[tscalar]])
        self.check_lshape(jt2, [[tscalar]])
        self.assertEqual([s for s in jt2.rshape], [s for s in tscalar.shape])
        self.assertEqual(jt2.eshape, [])

        jt3 = jt2[0]
        self.check_lshape(jt3, [tscalar])
        self.assertEqual([s for s in jt3.rshape], [s for s in tscalar.shape])
        self.assertEqual(jt3.eshape, [])

        jt3 = jt2[0:1]
        self.check_lshape(jt3, [[tscalar]])
        self.assertEqual([s for s in jt3.rshape], [s for s in tscalar.shape])
        self.assertEqual(jt3.eshape, [])

        jt3 = jt2[0:0]
        self.check_lshape(jt3, [])

        with self.assertRaises(IndexError):
            jt3 = jt2[1]

        tempty = torch.tensor([]).to(device).to(dtype)
        jt4 = fvdb.JaggedTensor([tempty])
        self.assertEqual([s for s in jt4.rshape], [s for s in tempty.shape])
        self.check_lshape(jt4, [tempty])
        self.assertEqual(jt4.eshape, [])

        jt5 = fvdb.JaggedTensor([[tempty]])
        self.check_lshape(jt5, [[tempty]])
        self.assertEqual([s for s in jt5.rshape], [s for s in tempty.shape])
        self.assertEqual(jt5.eshape, [])

        ts2 = tscalar.unsqueeze(0)
        jt6 = fvdb.JaggedTensor([ts2])
        self.check_lshape(jt6, [ts2])
        self.assertEqual(jt6.eshape, [1])

    @parameterized.expand(all_device_dtype_combos)
    def test_jagged_create(self, device, dtype):
        def buildit1(lsizes, eshape):
            ts = []
            for s in lsizes:
                tshape = [s] + list(eshape)
                ts.append(torch.randn(tshape, device=device, dtype=dtype))
            return ts, fvdb.JaggedTensor(ts)

        def buildit2(lsizes, eshape):
            ts = []
            for li in lsizes:
                ts.append([])
                for s in li:
                    tshape = [s] + list(eshape)
                    ts[-1].append(torch.randn(tshape, device=device, dtype=dtype))
            return ts, fvdb.JaggedTensor(ts)

        lsizes = [10, 20, 30, 40, 50]
        esizes = [(), (3,), (3, 4), (3, 4, 5)]
        funcs = [fvdb.jrand, fvdb.jrandn, fvdb.jzeros, fvdb.jones, fvdb.jempty]
        for func in funcs:
            for eshape in esizes:
                for rgrad in [True, False]:
                    jt = func(lsizes, eshape, device=device, dtype=dtype, requires_grad=rgrad)
                    self.assertEqual(rgrad, jt.requires_grad)
                    l2, jt2 = buildit1(lsizes, eshape)
                    self.check_lshape(jt2, l2)
                    self.check_lshape(jt, l2)
                    self.assertEqual(jt.device.type, torch.device(device).type)
                    self.assertEqual(jt.dtype, dtype)
                    self.assertEqual(jt.lshape, lsizes)
                    self.assertEqual(tuple(jt.eshape), eshape)

        lsizes = [[10, 20, 30, 40, 50], [100, 200], [1], [0, 0, 4]]
        esizes = [(), (3,), (3, 4), (3, 4, 5)]
        funcs = [fvdb.jrand, fvdb.jrandn, fvdb.jzeros, fvdb.jones, fvdb.jempty]
        for func in funcs:
            for eshape in esizes:
                for rgrad in [True, False]:
                    jt = func(lsizes, eshape, device=device, dtype=dtype, requires_grad=rgrad)
                    self.assertEqual(rgrad, jt.requires_grad)
                    l2, jt2 = buildit2(lsizes, eshape)
                    self.check_lshape(jt2, l2)
                    self.check_lshape(jt, l2)
                    self.assertEqual(jt.device.type, torch.device(device).type)
                    self.assertEqual(jt.dtype, dtype)
                    self.assertEqual(jt.lshape, lsizes)
                    self.assertEqual(tuple(jt.eshape), eshape)

        for func in funcs:
            lsizes = []
            eshape = (3, 4)
            with self.assertRaises(ValueError):
                jt = fvdb.jrand(lsizes, eshape, device=device, dtype=dtype)

            lsizes = [[]]
            with self.assertRaises(ValueError):
                jt = fvdb.jrand(lsizes, eshape, device=device, dtype=dtype)

            lsizes = [[], []]
            with self.assertRaises(ValueError):
                jt = fvdb.jrand(lsizes, eshape, device=device, dtype=dtype)

    @parameterized.expand(all_device_dtype_combos)
    def test_assignment(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt1 = jt1 + 1.0
        self.assertTrue(not torch.all(jt1.jdata == jt2.jdata).item())

        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 += 1.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 += 1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 += jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 -= 1.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 -= 1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 -= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 *= 3.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 *= 3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 *= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 /= 2.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 /= 2
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 /= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 %= 2.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 //= 2
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 //= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 //= 2.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 //= 2
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 //= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 %= 3.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 %= 3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 %= jt3
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1
        jt3 = jt1.clone()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        jt2 = jt1
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())

        jt1 **= 2.0
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())
        jt1 **= 2
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1.jdata = jt1.jdata.abs()
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

        jt1 **= jt3.jdata.abs()  # Because NaN
        self.assertTrue(torch.all(jt1.jdata == jt2.jdata).item())
        self.assertTrue(not torch.all(jt1.jdata == jt3.jdata).item())

    @parameterized.expand(all_device_dtype_combos)
    def test_sqrt(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt1 = jt1.abs()
        jt2 = jt1.sqrt()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata.sqrt()))

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt1.abs_()
        jt2 = jt1.sqrt()
        jt1.sqrt_()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_floor(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.floor()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata.floor()))

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt1.abs_()
        jt2 = jt1.floor()
        jt1.floor_()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_ceil(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.ceil()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata.ceil()))

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt1.abs_()
        jt2 = jt1.ceil()
        jt1.ceil_()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_abs(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.abs()
        self.assertTrue(torch.all(jt2.jdata == jt1.jdata.abs()).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.abs()
        jt1.abs_()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata))

    @parameterized.expand(all_device_dtype_combos)
    def test_round(self, device, dtype):
        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.round()
        self.assertTrue(torch.all(jt2.jdata == jt1.jdata.round()).item())

        jt1 = fvdb.jrandn([[10, 20, 30], [40, 50, 60, 70], [80, 90]], (3, 4), device=device, dtype=dtype)
        jt2 = jt1.round()
        jt1.round_()
        self.assertTrue(torch.allclose(jt2.jdata, jt1.jdata))

    # def test_argsort(self):
    #     data = [torch.randn(np.random.randint(1024, 2048),) for _ in range(7)]
    #     jt = fvdb.JaggedTensor(data)
    #     idx = jt.jagged_argsort()

    #     pmt = jt.jdata[idx.jdata]
    #     jt_s = jt.jagged_like(pmt)
    #     for i in range(len(jt)):
    #         data_sorted, _ = torch.sort(data[i])
    #         self.assertTrue(torch.all(data_sorted == jt_s[i].jdata).item())
    #         self.assertTrue(torch.all(data_sorted == jt[i].jdata[idx[i].jdata]).item())


if __name__ == "__main__":
    unittest.main()
