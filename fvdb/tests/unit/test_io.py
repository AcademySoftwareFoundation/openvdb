import itertools
import os
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

import fvdb

from .common import random_drop_points_if_mutable


standard_dtypes_and_dims = [
    (torch.float16, 1),
    (torch.float16, ()),
    (torch.float32, 1),
    (torch.float32, 3),
    (torch.float32, 4),
    (torch.float32, ()),
    (torch.float64, 1),
    (torch.float64, 3),
    (torch.float64, 4),
    (torch.float64, ()),
    (torch.int32, 1),
    (torch.int32, ()),
    (torch.int64, 1),
    (torch.int64, ()),
    (torch.uint8, 4),
]

other_dtypes_and_dims = [
    (torch.uint8, 1),
    (torch.uint8, ()),
    (torch.uint8, (1, 4)),
    (torch.float32, (4, 3, 2, 1, 1, 1)),
    (torch.float16, (4, 4)),
    (torch.int32, (4, 7, 1)),
    (torch.int64, (1, 1, 1, 1, 1)),
    (torch.float64, (1, 3, 1, 3, 1, 3, 1)),
]

all_dtypes_and_dims = standard_dtypes_and_dims + other_dtypes_and_dims

all_names_compressed_devices_dtypes_and_dims = list(itertools.product(
    [1, 2, 4],        # Batch size
    [True, False],    # Include names or not
    [True, False],    # Compressed
    ['cpu', 'cuda'],  # Device
    [True, False],    # Mutable
    all_dtypes_and_dims))


class TestIO(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_names_compressed_devices_dtypes_and_dims)
    def test_save_and_load(self, batch_size, include_names, compressed, device, mutable, dtype_and_dim):
        dtype, dim = dtype_and_dim
        dim = [dim] if isinstance(dim, int) else list(dim)
        torch.manual_seed(0)
        np.random.seed(0)

        names = None
        if include_names:
            names = [f'grid-{i}' for i in range(batch_size)]
            if batch_size == 1 and np.random.rand() > 0.5:
                names = names[0]

        sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
        grid_ijk = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_ijk(grid_ijk, mutable=mutable)
        random_drop_points_if_mutable(grid, 0.5)
        sizes = [[int(grid.num_voxels[i].item())] + dim for i in range(batch_size)]
        data = fvdb.JaggedTensor(
            [(torch.rand(*sizes[i], device=device) * 256).to(dtype) for i in range(batch_size)])

        with tempfile.NamedTemporaryFile() as temp:
            fvdb.save(temp.name, grid, data, names=names, compressed=compressed)
            grid2, data2, names2 = fvdb.load(temp.name, device=device)

            if isinstance(names, str):
                names = [names] * batch_size
            elif names is None:
                names = [''] * batch_size

            self.assertTrue(names == names2, f"{names}, {names2}")

            # NOTE: (@fwilliams) For some reason, when we build on the GPU, things get reordered when we load.
            # This is due to a bug in NanoVDB. We can't test for equality, but we can test for equivalence.
            if device == 'cuda':
                for bi in range(batch_size):
                    grid_ijk_i = grid.ijk[bi].jdata
                    grid2_ijk_i = grid2.ijk[bi].jdata
                    grid_data_i = data[bi].jdata.unsqueeze(-1) if data.jdata.ndim == 1 else data[bi].jdata
                    grid2_data_i = data2[bi].jdata.unsqueeze(-1) if data2.jdata.ndim == 1 else data2[bi].jdata

                    grid_ijk_i_list = [tuple(grid_ijk_i[i].cpu().numpy().tolist()) for i in range(grid_ijk_i.shape[0])]
                    grid2_ijk_i_list = [tuple(grid2_ijk_i[i].cpu().numpy().tolist()) for i in range(grid2_data_i.shape[0])]

                    grid_data_i_list = [tuple(grid_data_i[i].view(-1).cpu().numpy().tolist()) for i in range(grid_data_i.shape[0])]
                    grid2_data_i_list = [tuple(grid_data_i[i].view(-1).cpu().numpy().tolist()) for i in range(grid2_data_i.shape[0])]

                    grid_i_dict = dict(zip(grid_ijk_i_list, grid_data_i_list))
                    grid2_i_dict = dict(zip(grid2_ijk_i_list, grid2_data_i_list))

                    self.assertTrue(set(grid_ijk_i_list) == set(grid2_ijk_i_list))
                    self.assertTrue(set(grid_data_i_list) == set(grid2_data_i_list))

                    # FIXME: (@fwilliams) -- This will fail because of a bug in NanoVDB
                    # self.assertTrue(grid_i_dict == grid2_i_dict)
            else:
                self.assertTrue(torch.all(grid.ijk.jdata == grid2.ijk.jdata))
                self.assertTrue(torch.all(data.jdata == data2.jdata))
                self.assertTrue(torch.all(grid.enabled_mask.jdata == grid2.enabled_mask.jdata))


    @parameterized.expand(itertools.product(['cuda', 'cpu'], [1, 3]))
    def test_save_and_load_without_data(self, device, batch_size):
        torch.manual_seed(0)
        np.random.seed(0)

        names = None

        sizes = [np.random.randint(10, 20) for _ in range(batch_size)]
        grid_ijk = fvdb.JaggedTensor(
            [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
        grid = fvdb.sparse_grid_from_ijk(grid_ijk)


        with tempfile.NamedTemporaryFile() as temp:
            fvdb.save(temp.name, grid, names=names, compressed=True)
            grid2, data2, names2 = fvdb.load(temp.name, device=device)

            if isinstance(names, str):
                names = [names] * batch_size
            elif names is None:
                names = [''] * batch_size

            self.assertTrue(names == names2, f"{names}, {names2}")

            for bi in range(batch_size):
                self.assertEqual(data2.jdata.numel(), 0)
                self.assertTrue(data2.joffsets.shape[0] == batch_size)
                self.assertTrue(data2.joffsets[bi][0].item() == 0)
                self.assertTrue(data2.joffsets[bi][1].item() == 0)
                self.assertTrue(data2.jidx.numel() == 0)

            # NOTE: (@fwilliams) For some reason, when we build on the GPU, things get reordered when we load.
            # This is due to a bug in NanoVDB. We can't test for equality, but we can test for equivalence.
            if device == 'cuda':
                for bi in range(batch_size):
                    grid_ijk_i = grid.ijk[bi].jdata
                    grid2_ijk_i = grid2.ijk[bi].jdata

                    grid_ijk_i_list = [tuple(grid_ijk_i[i].cpu().numpy().tolist()) for i in range(grid_ijk_i.shape[0])]
                    grid2_ijk_i_list = [tuple(grid2_ijk_i[i].cpu().numpy().tolist()) for i in range(grid2_ijk_i.shape[0])]
                    self.assertTrue(set(grid_ijk_i_list) == set(grid2_ijk_i_list))

                    # FIXME: (@fwilliams) -- This will fail because of a bug in NanoVDB
                    # self.assertTrue(grid_i_dict == grid2_i_dict)
            else:
                self.assertTrue(torch.all(grid.ijk.jdata == grid2.ijk.jdata))

    def test_load_basic(self):
        datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
        grids, data, names = fvdb.load(os.path.join(datadir, 'batch.nvdb'))
        grids, data, names = fvdb.load(os.path.join(datadir, 'smoke-blosc.nvdb'))

    @parameterized.expand(['cpu', 'cuda'])
    def test_name_too_long_raises(self, device):
        for batch_size in (1, 3):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            with self.assertRaises(ValueError):
                fvdb.save('temp.nvdb', grid, grid_ijk, compressed=True, names=['a'*1000]*batch_size)

    @parameterized.expand(['cpu', 'cuda'])
    def test_bad_length_raises(self, device):
        for batch_size in (1, 4):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            grid_data = fvdb.JaggedTensor(
                [torch.rand(int(grid.num_voxels[i].item()) + (-1)**(i%2)*2).to(device) for i in range(batch_size)])
            if batch_size % 2 == 0:
                self.assertEqual(grid_data.jdata.shape[0], grid.total_voxels)
            with tempfile.NamedTemporaryFile() as temp:
                with self.assertRaises(ValueError):
                    fvdb.save(temp.name, grid, grid_data, compressed=True, names=['a']*batch_size)

            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            grid_data = fvdb.JaggedTensor(
                [torch.rand(int(grid.num_voxels[i].item()) + (-1)**(i%2)*2, 3, 3).to(device) for i in range(batch_size)])
            if batch_size % 2 == 0:
                self.assertEqual(grid_data.jdata.shape[0], grid.total_voxels)
            with tempfile.NamedTemporaryFile() as temp:
                with self.assertRaises(ValueError):
                    fvdb.save(temp.name, grid, grid_data, compressed=True, names=['a']*batch_size)

    @parameterized.expand(['cpu', 'cuda'])
    def test_bad_device_raises(self, device):
        bad_device = 'cpu' if device == 'cuda' else 'cuda'
        for batch_size in (1, 4):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            grid_data = fvdb.JaggedTensor(
                [torch.rand(int(grid.num_voxels[i].item())).to(bad_device) for i in range(batch_size)])
            with self.assertRaises(ValueError):
                fvdb.save('temp.nvdb', grid, grid_data, compressed=True, names=['a']*batch_size)

    @parameterized.expand(['cpu', 'cuda'])
    def test_bad_names_raises(self, device):
        for batch_size in (1, 4):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            grid_data = fvdb.JaggedTensor(
                [torch.rand(int(grid.num_voxels[i].item())).to(device) for i in range(batch_size)])
            names = ['aaa'] * (batch_size + 1)
            with tempfile.NamedTemporaryFile() as temp:
                with self.assertRaises(ValueError):
                    fvdb.save(temp.name, grid, grid_data, compressed=True, names=names)

    @parameterized.expand(['cpu', 'cuda'])
    def test_nonexistent_name_raises(self, device):
        for batch_size in (1, 3):
            sizes = [np.random.randint(100, 200) for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            # data = fvdb.JaggedTensor([torch.rand(grid.num_voxels[i].item()).to(device) for i in range(batch_size)])
            with tempfile.NamedTemporaryFile() as temp:
                fvdb.save(temp.name, grid, compressed=True, names=[f'a_{i}' for i in range(batch_size)])

                with self.assertRaises(IndexError):
                    fvdb.load(temp.name, device=device, grid_id=['a_0', 'b', 'a_1', 'a_0'])

                with self.assertRaises(IndexError):
                    fvdb.load(temp.name, device=device, grid_id='c')

    @parameterized.expand(['cpu', 'cuda'])
    def test_one_voxel_grids(self, device):
        for batch_size in (1, 3):
            sizes = [1 for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            data = fvdb.JaggedTensor([torch.rand(1).squeeze().to(device)] * batch_size)
            with tempfile.NamedTemporaryFile() as temp:
                fvdb.save(temp.name, grid, data, compressed=False)
                grid2, data2, names = fvdb.load(temp.name, device=device)
                self.assertTrue(torch.all(grid2.ijk.jdata == grid.ijk.jdata))
                self.assertTrue(torch.all(data2.jdata == data.jdata))

            sizes = [1 for _ in range(batch_size)]
            grid_ijk = fvdb.JaggedTensor(
                [torch.randint(-512, 512, (sizes[i], 3)) for i in range(batch_size)]).to(device)
            grid = fvdb.sparse_grid_from_ijk(grid_ijk)
            data = fvdb.JaggedTensor([torch.rand(1).unsqueeze(-1).unsqueeze(-1).to(device)] * batch_size)
            with tempfile.NamedTemporaryFile() as temp:
                fvdb.save(temp.name, grid, data, compressed=False)
                grid2, data2, names = fvdb.load(temp.name, device=device)
                self.assertTrue(torch.all(grid2.ijk.jdata == grid.ijk.jdata))
                self.assertTrue(torch.all(data2.jdata == data.jdata))

    @parameterized.expand(['cpu', 'cuda'])
    def test_voxelsize_and_origin(self, device):
        torch.manual_seed(0)
        np.random.seed(0)

        pts = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],], device=device)
        test_grid = fvdb.sparse_grid_from_points(pts,
                                                 voxel_sizes=np.random.random()+0.00001,
                                                 origins=[np.random.randint(-100,100) for _ in range(3)])

        with tempfile.NamedTemporaryFile() as temp:
            fvdb.save(temp.name, test_grid)
            test_grid_from_file, _, _ = fvdb.load(temp.name, device=device)

            self.assertTrue(torch.all(test_grid.voxel_sizes == test_grid_from_file.voxel_sizes))
            self.assertTrue(torch.all(test_grid.origins == test_grid_from_file.origins))

if __name__ == '__main__':
    unittest.main()
