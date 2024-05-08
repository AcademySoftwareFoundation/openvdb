import unittest

import torch
from parameterized import parameterized

from fvdb import GridBatch

all_device_combos = [
    ['cpu', True],
    ['cuda', True],
    ['cpu', False],
    ['cuda', False],
]

class TestRayMarching(unittest.TestCase):
    def setUp(self):
        pass

    @parameterized.expand(all_device_combos)
    def test_world_to_dual(self, device, include_end_segments: bool):
        grid = GridBatch(device=device)
        grid.set_from_dense_grid(1, [2, 2, 2])

        rays_o = torch.tensor([[-0.6, 0., 0.]], device=device)
        rays_d = torch.tensor([[1., 0., 0.]], device=device)
        nears = torch.tensor([0.], device=device)
        fars = torch.tensor([5.], device=device)
        step_size = 0.4
        cone_angle = 0.0
        if include_end_segments:
            t_targets = torch.tensor([0.1, 0.4, 0.8, 1.2, 1.6, 2.0, 2.1], device=device)
        else:
            t_targets = torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], device=device)

        _, indices, intervals = grid.uniform_ray_samples(
            rays_o, rays_d, nears, fars, step_size, cone_angle, include_end_segments
        )
        t_starts, t_ends = torch.unbind(intervals.jdata, dim=-1)
        ray_indices = indices.jdata.long()

        assert torch.allclose(t_starts, t_targets[:-1])
        assert torch.allclose(t_ends, t_targets[1:])
        assert torch.allclose(ray_indices, torch.zeros_like(ray_indices))



if __name__ == '__main__':
    unittest.main()
