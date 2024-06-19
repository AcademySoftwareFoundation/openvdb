import torch
import torch_scatter

import fvdb

from .abc import BaseBackend
from .hash_table import torch_unique

print("SparseFeatureHierarchy Backend: fVDB 0.0.0")


class SparseFeatureHierarchy(BaseBackend):

    CONFORM_OFFSETS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    def __init__(self, depth: int, voxel_size: float, device, range_kernel):
        super().__init__(depth, voxel_size, device, range_kernel)

        self._depth = depth
        self._voxel_size = voxel_size
        self._device = device
        self._range_kernel = range_kernel
        self._vox_sizes = [voxel_size * (2 ** d) for d in range(depth)]
        self._indexes = [fvdb.GridBatch(device=device) for d in range(self.depth)]

    @property
    def depth(self):
        return self._depth

    @property
    def voxel_size(self):
        return self._indexes[0].voxel_sizes[0]

    def get_stride(self, depth: int):
        return 2 ** depth

    def get_coords(self, depth: int, expand: int = 0, conforming: bool = False):
        scale = 2 ** depth
        if self._indexes[depth].total_voxels == 0:
            return torch.zeros(0, 3, device=self._device, dtype=torch.int32)

        base_coords = self._indexes[depth].ijk.jdata.int()

        if expand >= 3:
            mc_offsets = self._range_kernel()(expand) * scale
            base_coords = (base_coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
                           mc_offsets.unsqueeze(0)).view(-1, 3)
            base_coords = torch_unique(base_coords, dim=0)

        if conforming:
            base_coords = (base_coords / scale / 2.).floor().int() * scale * 2
            base_coords = torch_unique(base_coords, dim=0)
            conform_offsets = torch.tensor(
                self.CONFORM_OFFSETS, dtype=torch.int32, device=base_coords.device) * scale
            base_coords = (base_coords.unsqueeze(dim=1).repeat(1, 8, 1) +
                           conform_offsets.unsqueeze(0)).view(-1, 3)

        return base_coords

    def get_num_voxels(self, depth: int):
        return self._indexes[depth].total_voxels

    def get_voxel_centers(self, depth: int, normalized: bool = False):
        return (self.get_coords(depth) + 2 ** depth / 2.) * \
               (self._voxel_size if not normalized else 1.0)

    def __repr__(self):
        return "fVDB"

    def get_coords_neighbours(self, source_coords: torch.Tensor,
                              source_stride: int, target_depth: int,
                              nn_kernel: torch.Tensor, conv_based: bool = False,
                              transposed: bool = False, raw: bool = False):
        assert 0 <= target_depth < self._depth

        target_stride = 2 ** target_depth
        if not conv_based:
            # Flaw: If the layers are different (source stride < target stride), you may end up with
            #   neighbours that has no overlap support.
            assert source_stride <= target_stride, "Data must be deeper and has more nodes."
            # Compute voxel center offsets.
            quantized_source_coords = torch.div(
                source_coords.detach() + 0.5 * source_stride, target_stride,
                rounding_mode='floor').int() * target_stride
            c_offset = (quantized_source_coords - source_coords) / source_stride + \
                       (target_stride // source_stride - 1) / 2.
        else:
            assert not source_coords.requires_grad
            assert source_stride >= target_stride, "Data must be sparser and shallower."
            quantized_source_coords = source_coords

        # (N, 3) x (K, 3) -> (K, N, 3)
        queried_coords = quantized_source_coords.unsqueeze(0) + \
                         (nn_kernel * 2 ** target_depth).unsqueeze(1)
        hash_res = self._indexes[target_depth].ijk_to_index(queried_coords.reshape(-1, 3))
        hash_res = hash_res.jdata.reshape(-1, quantized_source_coords.size(0))

        if transposed:
            hash_res = hash_res.T

        if raw:
            return hash_res

        nbsizes = torch.sum(hash_res != -1, dim=1)

        if transposed:
            source_ids, kernel_ids = torch.where(hash_res != -1)
            target_ids = hash_res[source_ids, kernel_ids]
        else:
            kernel_ids, source_ids = torch.where(hash_res != -1)
            target_ids = hash_res[kernel_ids, source_ids]

        neighbour_types = nn_kernel[kernel_ids]

        if not conv_based:
            neighbour_types = neighbour_types.float()
            neighbour_types *= 2 ** target_depth / source_stride
            neighbour_types += c_offset[source_ids, :3]

        return source_ids, target_ids, neighbour_types, nbsizes

    def get_self_neighbours(self, source_depth: int, target_depth: int, target_range: int,
                            conv_based: bool = False):
        assert 0 <= source_depth < self.depth and 0 <= target_depth < self.depth

        # conv_based flag will be ignored if source-depth == target-depth, because this is anyway
        # covered in both situations.
        inv_op = False
        if not conv_based and source_depth != target_depth:
            # In the case where source is shallower/fewer than target, we inverse the operation
            if source_depth > target_depth:
                source_depth, target_depth, inv_op = target_depth, source_depth, True

        def recover_inv_op(inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs):
            if not inv_op:
                return inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs
            else:
                near_mask = torch.all(inv_nts.abs() < target_range / 2. + 1.0e-6, dim=1)
                inv_nts = -inv_nts * 2 ** (source_depth - target_depth)
                return inv_tgt_ids[near_mask], inv_src_ids[near_mask], inv_nts[near_mask], None

        # Only compute incremental part:
        neighbour_kernel = self._range_kernel()(target_range)
        source_ids, target_ids, neighbour_types, nbsizes = self.get_coords_neighbours(
            self._indexes[source_depth].ijk.jdata,
            2 ** source_depth,
            target_depth, neighbour_kernel, conv_based
        )

        return recover_inv_op(source_ids, target_ids, neighbour_types, nbsizes)

    def evaluate_voxel_status(self, coords: torch.Tensor, depth: int):
        raise NotImplementedError

    def split_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor):
        raise NotImplementedError

    def _trilinear_weights(self, xyz: torch.Tensor, tree_stride: int, xyz_data: torch.Tensor = 1,
                           compute_grad: bool = False):
        # Gradient is alpha_data w.r.t. xyz.
        q_coords = xyz / self._voxel_size
        d_coords = (q_coords / tree_stride).floor() * tree_stride
        rel_coords = q_coords - d_coords - tree_stride / 2.
        oct_sign = torch.sign(rel_coords)
        oct_local = torch.abs(rel_coords) / tree_stride

        alpha_coords = []
        alpha_data = []
        grad_alpha_data = []
        for nx, ny, nz in self.CONFORM_OFFSETS:
            alpha_coords.append((d_coords + torch.stack([nx * oct_sign[:, 0],
                                                         ny * oct_sign[:, 1],
                                                         nz * oct_sign[:, 2]],
                                                        dim=1) * tree_stride).int())
            alpha_x = oct_local[:, 0] if nx == 1 else 1 - oct_local[:, 0]
            alpha_y = oct_local[:, 1] if ny == 1 else 1 - oct_local[:, 1]
            alpha_z = oct_local[:, 2] if nz == 1 else 1 - oct_local[:, 2]
            alpha_os = alpha_x * alpha_y * alpha_z

            if compute_grad:
                assert xyz_data == 1, "What do you want?"
                d_alpha_x = (oct_sign[:, 0] if nx == 1 else -oct_sign[:, 0]) / (self._voxel_size * tree_stride)
                d_alpha_y = (oct_sign[:, 1] if ny == 1 else -oct_sign[:, 1]) / (self._voxel_size * tree_stride)
                d_alpha_z = (oct_sign[:, 2] if nz == 1 else -oct_sign[:, 2]) / (self._voxel_size * tree_stride)
                grad_alpha_data.append(torch.stack([
                    d_alpha_x * alpha_y * alpha_z,
                    alpha_x * d_alpha_y * alpha_z,
                    alpha_x * alpha_y * d_alpha_z
                ], dim=1))

            alpha_data.append(alpha_os * xyz_data if isinstance(xyz_data, int) or xyz_data.ndim == 1 else
                              alpha_os[:, None] * xyz_data)
        alpha_coords = torch.cat(alpha_coords, dim=0)
        alpha_data = torch.cat(alpha_data, dim=0)

        if compute_grad:
            return alpha_coords, alpha_data, torch.cat(grad_alpha_data, dim=0)

        return alpha_coords, alpha_data

    def _identity_kernel(self):
        return torch.tensor([[0, 0, 0]], dtype=torch.int32, device=self._device)

    def splat_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor = None,
                   check_corr: bool = True, return_nf_mask: bool = False):
        """
        Splat the data onto the tree with tri-linear interpolation.
        :param xyz: data position
        :param data_depth: depth of the octree to splat onto.
        :param data: (N,) or (N,C) None means all ones, weight should be pre-multiplied to data if applicable
        :return: (V,) or (V,C).
        """
        if data is not None:
            assert data.size(0) == xyz.size(0), "Input data must agree with xyz in size."
        else:
            data = 1

        tree_stride = 2 ** data_depth
        alpha_coords, alpha_data = self._trilinear_weights(xyz, tree_stride, data)

        # align normal_coords and tree_coords.
        alpha_source, alpha_target, _, nb_sizes = self.get_coords_neighbours(
            alpha_coords, tree_stride, data_depth, self._identity_kernel(), transposed=True)

        # Make sure that each query coordinates has one correspondent:
        if alpha_source.size(0) < alpha_coords.size(0) and check_corr:
            print("Warning: Some grids that normal should be splatted onto is missing because expansion is too small. "
                  f"# Should = {alpha_coords.size(0)}, Actual = {alpha_source.size(0)}.")
        splat_res = torch_scatter.scatter_sum(alpha_data[alpha_source], alpha_target, dim=0,
                                              dim_size=self.get_num_voxels(data_depth))
        if return_nf_mask:
            # If a point can only be splatted on to less than 4 voxels, it is a bad splat.
            return splat_res, nb_sizes.reshape(8, -1).sum(0) < 4
        return splat_res

    def build_hierarchy_dense(self, xyz: torch.Tensor, expand_range: int = 0):
        raise NotImplementedError

    def build_hierarchy_subdivide(self, xyz: torch.Tensor, subdivide_policy, expand: bool = False,
                                  limit_adaptive_depth: int = 100, **policy_kwargs):
        raise NotImplementedError

    def build_hierarchy_adaptive(self, xyz: torch.Tensor, xyz_density: torch.Tensor, log_base: float = 4.0,
                                 min_density: float = 8.0,
                                 limit_adaptive_depth: int = 100):
        raise NotImplementedError

    def update_coords(self, depth: int, coords: torch.Tensor):
        if coords is None:
            return
        assert coords.ndim == 2 and coords.size(1) == 3, coords.size()
        self._indexes[depth].set_from_ijk(coords, [0, 0, 0], [0, 0, 0], voxel_sizes=self._vox_sizes[depth])
        coords_idx = self._indexes[depth].ijk_to_index(coords)
        permutation = torch.empty(coords.size(0), dtype=torch.long, device=self._device)
        permutation[coords_idx.jdata] = torch.arange(coords.size(0), dtype=torch.long, device=self._device)
        return coords[permutation], permutation

    def trilinear_interpolate(self, queries: torch.Tensor, depth: int, feature: torch.Tensor,
                              feature_bg: torch.Tensor = None, compute_grad: bool = False):
        raise NotImplementedError


if __name__ == '__main__':
    pass
