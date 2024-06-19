from typing import List, Union

import numpy as np
import torch
import torch_scatter

from ..ext import CuckooHashTable
from .abc import BaseBackend

print("SparseFeatureHierarchy Backend: Hash Table")


def torch_unique(input: torch.Tensor, sorted: bool = False, return_inverse: bool = False,
                 return_counts: bool = False, dim: int = None):
    """
    If used with dim, then torch.unique will return a flattened tensor. This fixes that behaviour.
    :param input: (Tensor) – the input tensor
    :param sorted: (bool) – Whether to sort the unique elements in ascending order before returning as output.
    :param return_inverse: (bool) – Whether to also return the indices for where elements in the original input
        ended up in the returned unique list.
    :param return_counts: (bool) – Whether to also return the counts for each unique element.
    :param dim: (int) – the dimension to apply unique. If None, the unique of the flattened input is returned.
        default: None
    :return: output, inverse_indices, counts
    """
    res = torch.unique(input, sorted, return_inverse, return_counts, dim)

    if dim is not None and input.size(dim) == 0:
        output_size = list(input.size())
        output_size[dim] = 0
        if isinstance(res, torch.Tensor):
            res = res.reshape(output_size)
        else:
            res = list(res)
            res[0] = res[0].reshape(output_size)

    return res


class NeighbourMaps:
    """
    A cache similar to kernel map, without the need of re-computing everything when enlarging neighbourhoods.
    """

    def __init__(self, device):
        # Cached maps (src-depth, tgt-depth) -> (tgt-neighbour-size 1,3,5, src-id, tgt-id, neighbour-types, nbsizes)
        #   Note: none of the relevant range here is in strided format!
        self.cache = {}
        self.device = device

    def get_map(self, source_depth: int, target_depth: int, target_range: int, force_recompute: bool = False):
        """
        Given the query, return the existing part and also the part needed to be queried.
        :return: tuple (src-id, tgt-id, neighbour-types, nbsizes, ranges lacked [a,b] )
        """
        if (source_depth, target_depth) in self.cache.keys():
            if force_recompute:
                del self.cache[(source_depth, target_depth)]
                max_range, exist_src, exist_tgt, exist_nt, exist_nbs = -1, None, None, None, None
            else:
                max_range, exist_src, exist_tgt, exist_nt, exist_nbs = self.cache[(source_depth, target_depth)]
        else:
            max_range, exist_src, exist_tgt, exist_nt, exist_nbs = -1, None, None, None, None

        if target_range == max_range:
            return exist_src, exist_tgt, exist_nt, exist_nbs, None
        elif target_range < max_range:
            tr3 = target_range * target_range * target_range
            n_query = torch.sum(exist_nbs[:tr3])
            return exist_src[:n_query], exist_tgt[:n_query], exist_nt[:n_query], exist_nbs[:tr3], None
        else:
            return exist_src, exist_tgt, exist_nt, exist_nbs, [max_range + 2, target_range]

    def update_map(self, source_depth: int, target_depth: int, target_range: int, res: list):
        self.cache[(source_depth, target_depth)] = [target_range] + res


class SparseFeatureHierarchy(BaseBackend):

    CONFORM_OFFSETS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    def __init__(self, depth: int, voxel_size: float, device, range_kernel):
        self._depth = depth
        self._voxel_size = voxel_size
        self._device = device
        self._range_kernel = range_kernel

        # Conv-based include same-level
        self._conv_nmap = NeighbourMaps(self._device)
        # Region-based exclude same-level
        self._region_nmap = NeighbourMaps(self._device)

        self._strides = [2 ** d for d in range(self.depth)]
        # List of torch.Tensor (Nx3)
        self._coords = [None for d in range(self.depth)]
        self._hash_table: List[CuckooHashTable] = [None for d in range(self.depth)]

    @property
    def depth(self):
        return self._depth

    @property
    def voxel_size(self):
        return self._voxel_size

    def get_stride(self, depth: int):
        return self._strides[depth]

    def get_coords(self, depth: int, expand: int = 0, conforming: bool = False):
        scale = self._strides[depth]
        base_coords = self._coords[depth]

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
        return self._coords[depth].size(0) if self._coords[depth] is not None else 0

    def get_voxel_centers(self, depth: int, normalized: bool = False):
        return (self.get_coords(depth) + self._strides[depth] / 2.) * (self._voxel_size if not normalized else 1.0)

    def __repr__(self):
        stat = f"Depth={self.depth}:\n"
        for stride, coords in zip(self._strides, self._coords):
            if coords is None:
                stat += f" + [{stride}] Empty\n"
                continue
            c_min = torch.min(coords, dim=0).values
            c_max = torch.max(coords, dim=0).values
            stat += f" + [{stride}] #Voxels={coords.size(0)} " \
                    f"Bound=[{c_min[0]},{c_max[0]}]x[{c_min[1]},{c_max[1]}]x[{c_min[2]},{c_max[2]}]\n"
        return stat

    def _update_hash_table(self):
        for d in range(self.depth):
            self._hash_table[d] = CuckooHashTable(data=self._coords[d])
            assert self._hash_table[d].dim == 3

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

    def get_coords_neighbours(self, source_coords: torch.Tensor, source_stride: int, target_depth: int,
                              nn_kernel: torch.Tensor, conv_based: bool = False, transposed: bool = False, raw: bool = False):
        """
        A generic interface for querying neighbourhood information. (This is without cache)
            For all source (data), find all target whose neighbourhood (in target level) covers it,
        will also return the relative position of the two.
        :param nn_kernel: Unit is 1
        :param transposed: allows efficient per-source handling.
        """
        assert 0 <= target_depth < self._depth

        if not conv_based:
            # Flaw: If the layers are different (source stride < target stride), you may end up with
            #   neighbours that has no overlap support.
            assert source_stride <= self._strides[target_depth], "Data must be deeper and has more nodes."
            # Compute voxel center offsets.
            quantized_source_coords = torch.div(
                source_coords.detach() + 0.5 * source_stride, self._strides[target_depth],
                rounding_mode='floor').int() * self._strides[target_depth]
            c_offset = (quantized_source_coords - source_coords) / source_stride + \
                       (self._strides[target_depth] // source_stride - 1) / 2.
        else:
            assert not source_coords.requires_grad
            assert source_stride >= self._strides[target_depth], "Data must be sparser and shallower."
            quantized_source_coords = source_coords

        hash_res = self._hash_table[target_depth].query(
            quantized_source_coords, nn_kernel * self._strides[target_depth])  # (K, N)

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
            neighbour_types *= self._strides[target_depth] / source_stride
            neighbour_types += c_offset[source_ids, :3]

        return source_ids, target_ids, neighbour_types, nbsizes

    def get_self_neighbours(self, source_depth: int, target_depth: int, target_range: int,
                            conv_based: bool = False):
        """
        :param source_depth: source depth where you want the coord id to start from
        :param target_depth: target depth where you want the coord id to shoot to
        :param target_range: must be odd, logical neighbourhood range to search for, e.g. 5 for B2 basis.
        :return: [sid, tid]
        """
        assert 0 <= source_depth < self.depth and 0 <= target_depth < self.depth

        tree_coords, tree_strides = self._coords, self._strides

        # conv_based flag will be ignored if source-depth == target-depth, because this is anyway
        #   covered in both situations.
        inv_op = False
        if not conv_based and source_depth != target_depth:
            neighbour_maps = self._region_nmap
            # In the case where source is shallower/fewer than target, we inverse the operation
            if source_depth > target_depth:
                source_depth, target_depth, inv_op = target_depth, source_depth, True
        else:
            neighbour_maps = self._conv_nmap

        def recover_inv_op(inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs):
            if not inv_op:
                return inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs
            else:
                # Filter far away nodes.
                near_mask = torch.all(inv_nts.abs() < target_range / 2. + 1.0e-6, dim=1)
                # Convert back neighbour types.
                inv_nts = -inv_nts / tree_strides[target_depth] * tree_strides[source_depth]
                return inv_tgt_ids[near_mask], inv_src_ids[near_mask], inv_nts[near_mask], None

        exist_src, exist_tgt, exist_nt, exist_nbs, lack_range = \
            neighbour_maps.get_map(source_depth, target_depth, target_range)

        if lack_range is None:
            return recover_inv_op(exist_src, exist_tgt, exist_nt, exist_nbs)

        # Only compute incremental part:
        neighbour_kernel = self._range_kernel()(target_range)
        starting_lap = max(0, lack_range[0] - 2)
        starting_lap = starting_lap ** 3
        neighbour_kernel = neighbour_kernel[starting_lap:]

        source_ids, target_ids, neighbour_types, nbsizes = self.get_coords_neighbours(
            tree_coords[source_depth], tree_strides[source_depth], target_depth, neighbour_kernel, conv_based
        )

        if exist_src is not None:
            source_ids = torch.cat([exist_src, source_ids], dim=0)
            target_ids = torch.cat([exist_tgt, target_ids], dim=0)
            neighbour_types = torch.cat([exist_nt, neighbour_types], dim=0)
            nbsizes = torch.cat([exist_nbs, nbsizes], dim=0)

        # Cache result for future use.
        neighbour_maps.update_map(source_depth, target_depth, target_range,
                                  [source_ids, target_ids, neighbour_types, nbsizes])

        return recover_inv_op(source_ids, target_ids, neighbour_types, nbsizes)

    def evaluate_voxel_status(self, coords: torch.Tensor, depth: int):
        """
        Evaluate the voxel status of given coordinates
        :param coords: (N, 3)
        :param depth: int
        :return: (N, ) long tensor, with value 0,1,2
        """
        from core.hashtree import VoxelStatus
        status = torch.full((coords.size(0),), VoxelStatus.VS_NON_EXIST.value, dtype=torch.long, device=coords.device)
        sidx, _, _, _ = self.get_coords_neighbours(
            coords, self._strides[depth], depth, self._identity_kernel(), conv_based=True)
        status[sidx] = VoxelStatus.VS_EXIST_STOP.value

        if depth > 0:
            # Next level.
            conform_offsets = torch.tensor(self.CONFORM_OFFSETS, dtype=torch.int32, device=self._device) * \
                              self._strides[depth - 1]
            conform_coords = (coords[sidx].unsqueeze(dim=1).repeat(1, 8, 1) + conform_offsets.unsqueeze(0)).view(-1, 3)
            qidx, _, _, _ = self.get_coords_neighbours(
                conform_coords, self._strides[depth - 1], depth - 1, self._identity_kernel(), conv_based=True)
            qidx = torch.div(qidx, 8, rounding_mode='floor')
            status[sidx[qidx]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def split_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor):
        """
        Split the data from the tree to query positions, with tri-linear interpolations.
            This is the inverse operation of the splat function, used in decoders.
        :param xyz: query positions.
        :param data_depth: depth of the octree to split from
        :param data: (V, C).
        :return: (N, C)
        """
        tree_stride = self._strides[data_depth]
        assert data.size(0) == self._coords[data_depth].size(0), "Tree data does not agree on size."

        alpha_coords, alpha_weight = self._trilinear_weights(xyz, tree_stride)
        alpha_source, alpha_target, _, _ = self.get_coords_neighbours(
            alpha_coords, tree_stride, data_depth, self._identity_kernel())
        return torch_scatter.scatter_sum(data[alpha_target] * alpha_weight[alpha_source, None],
                                         alpha_source % xyz.size(0), dim=0,
                                         dim_size=xyz.size(0))

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

        tree_stride = self._strides[data_depth]
        alpha_coords, alpha_data = self._trilinear_weights(xyz, tree_stride, data)

        # align normal_coords and tree_coords.
        alpha_source, alpha_target, _, nb_sizes = self.get_coords_neighbours(
            alpha_coords, tree_stride, data_depth, self._identity_kernel(), transposed=True)

        # Make sure that each query coordinates has one correspondent:
        if alpha_source.size(0) < alpha_coords.size(0) and check_corr:
            print("Warning: Some grids that normal should be splatted onto is missing because expansion is too small. "
                  f"# Should = {alpha_coords.size(0)}, Actual = {alpha_source.size(0)}.")
        splat_res = torch_scatter.scatter_sum(alpha_data[alpha_source], alpha_target, dim=0,
                                         dim_size=self._coords[data_depth].size(0))
        if return_nf_mask:
            # If a point can only be splatted on to less than 4 voxels, it is a bad splat.
            return splat_res, nb_sizes.reshape(8, -1).sum(0) < 4
        return splat_res

    def _quantize_coords(self, xyz: torch.Tensor, data_depth: int):
        # Note this is just splat_data with NEW_BRANCH.
        tree_stride = self._strides[data_depth]
        alpha_coords, _ = self._trilinear_weights(xyz, tree_stride)
        alpha_coords = torch_unique(alpha_coords, dim=0)
        return alpha_coords

    def build_hierarchy_dense(self, xyz: torch.Tensor, expand_range: int = 0):
        """
        Rebuild the tree structure, based on current xyz, voxel_size and depth.
        """
        if expand_range == 2:
            unique_coords = self._quantize_coords(xyz, 0)
        else:
            coords = torch.div(xyz, self._voxel_size).floor().int()
            unique_coords = torch_unique(coords, dim=0)
            if expand_range > 0:
                offsets = self._range_kernel()(expand_range)
                my_pad = (unique_coords.unsqueeze(dim=1).repeat(1, offsets.size(0), 1) +
                          offsets.unsqueeze(0)).view(-1, 3)
                unique_coords = torch_unique(my_pad, dim=0)

        self._coords = [unique_coords]
        for d in range(1, self.depth):
            coords = torch.div(self._coords[-1], self._strides[d], rounding_mode='floor') * self._strides[d]
            coords = torch_unique(coords, dim=0)
            self._coords.append(coords)
        self._update_hash_table()

    def build_hierarchy_subdivide(self, xyz: torch.Tensor, subdivide_policy, expand: bool = False,
                                  limit_adaptive_depth: int = 100, **policy_kwargs):
        """
        Build a hierarchy, based on subdivision policy
        :return:
        """
        current_pts = xyz / self._voxel_size
        inv_mapping = None
        xyz_depth = torch.full((xyz.size(0),), fill_value=self._depth - 1, device=self._device, dtype=torch.int)
        xyz_depth_inds = torch.arange(xyz.size(0), device=self._device, dtype=torch.long)

        for d in range(self._depth - 1, -1, -1):
            if d != self._depth - 1:
                nxt_mask = subdivide_policy(current_pts, inv_mapping, **policy_kwargs)
                current_pts = current_pts[nxt_mask]
                xyz_depth_inds = xyz_depth_inds[nxt_mask]
                policy_kwargs = {k: v[nxt_mask] if isinstance(v, torch.Tensor) else v for k, v in policy_kwargs.items()}
                xyz_depth[xyz_depth_inds] -= 1
            coords = torch.div(current_pts, self.get_stride(d), rounding_mode='floor').int() * self._strides[d]
            unique_coords, inv_mapping = torch_unique(coords, dim=0, return_inverse=True)
            self._coords[d] = unique_coords
        xyz_depth.clamp_(max=limit_adaptive_depth - 1)

        if expand:
            self._coords = []
            for d in range(self.depth):
                depth_samples = xyz[xyz_depth <= d]
                coords = self._quantize_coords(depth_samples, d)
                if depth_samples.size(0) == 0:
                    print(f"-- disregard level {d} due to insufficient samples!")
                self._coords.append(coords)
        self._update_hash_table()

        return xyz_depth

    def build_hierarchy_adaptive(self, xyz: torch.Tensor, xyz_density: torch.Tensor, log_base: float = 4.0,
                                 min_density: float = 8.0,
                                 limit_adaptive_depth: int = 100):
        """
        Build a hierarchy similar to Adaptive-OCNN, i.e., finest voxels does not cover all surfaces,
            but only detailed parts. Coarse voxels, however, must cover all fine voxels.
            However, in a uniform dense sampling case, this falls back to the build_encoder_hierarchy_dense case.
        :param log_base: (float), used to determine how to split depths, the smaller, the more levels are going
            to be used. 4 is chosen in the original paper, which matches the 2-manifold structure.
        :param min_density: (float). The minimum normalized density (Unit: #points/voxel) to have for each point,
            so that when the density is smaller than this threshold, a coarser voxel is used for splatting this point.
            Any points with density larger than this threshold will be put to level-0.
            Note: This should be kept fixed most of the time because having too few samples within a voxel is bound
        :param limit_adaptive_depth: (int) depth limitation
        to fail and lead to holes. Tune voxel size or sub-sample point to get what you want.
        """
        # Compute expected depth.
        xyz_depth = -(torch.log(xyz_density / min_density) / np.log(log_base)).floor().int().clamp_(max=0)
        xyz_depth.clamp_(max=min(self.depth - 1, limit_adaptive_depth - 1))

        # self.xyz_depth = (self.xyz[:, 0] < 0.0).int()
        # self.xyz_density = torch.ones((self.xyz.size(0),), device=self.device)

        # Determine octants by splatting.
        self._coords = []
        for d in range(self.depth):
            depth_samples = xyz[xyz_depth <= d]
            coords = self._quantize_coords(depth_samples, d)
            # if depth_samples.size(0) == 0:
            #     print(f"-- disregard level {d} due to insufficient samples!")
            self._coords.append(coords)

        self._update_hash_table()
        return xyz_depth

    def update_coords(self, depth: int, coords: Union[torch.Tensor, None]):
        if coords is None:
            coords = torch.zeros((0, 3), dtype=torch.int32, device=self._device)
        assert coords.ndim == 2 and coords.size(1) == 3, coords.size()
        self._coords[depth] = coords
        self._hash_table[depth] = CuckooHashTable(data=self._coords[depth])
        return coords, torch.arange(coords.size(0), dtype=torch.long, device=coords.device)

    def _identity_kernel(self):
        return torch.tensor([[0, 0, 0]], dtype=torch.int32, device=self._device)

    def _trilerp_light(self, queries: torch.Tensor, depth: int, feature: torch.Tensor, compute_grad: bool = False):
        """
        This version use less memory...
        """
        alpha_res = self._trilinear_weights(queries, self._strides[depth], compute_grad=compute_grad)

        if compute_grad:
            alpha_coords, alpha_weight, grad_alpha_weight = alpha_res
        else:
            alpha_coords, alpha_weight = alpha_res

        # For the logic here refer to 'splat_data'
        alpha_source, alpha_target, _, nb_sizes = self.get_coords_neighbours(
            alpha_coords, self._strides[depth], depth, self._identity_kernel(), transposed=False)

        pts_source = alpha_source % queries.size(0)
        depth_feature = torch_scatter.scatter_sum(
            feature[alpha_target] * alpha_weight[alpha_source, None],
            pts_source, dim=0, dim_size=queries.size(0))

        if compute_grad:
            depth_grad = torch_scatter.scatter_sum(
                feature[alpha_target][:, :, None] * grad_alpha_weight[alpha_source, None, :],
                pts_source, dim=0, dim_size=queries.size(0)
            )
        else:
            depth_grad = None

        return depth_feature, depth_grad

    def trilinear_interpolate(self, queries: torch.Tensor, depth: int, feature: torch.Tensor,
                              feature_bg: torch.Tensor = None, compute_grad: bool = False):
        if feature_bg is not None:
            assert feature_bg.ndim == 1
            assert feature.size(1) == feature_bg.size(0), "Dimension not matched!"
        else:
            # Less memory version
            # return self._trilerp_light(queries, depth, feature, compute_grad)
            pass

        from ext import sparse_op
        nb_ids, nb_weight, nb_grad = sparse_op.trilerp(
            self._hash_table[depth].object,
            queries, self.voxel_size, self._strides[depth], compute_grad)

        nb_ids = nb_ids.view(-1)
        nb_weight = nb_weight.view(-1)
        pts_ids = torch.tile(torch.arange(queries.size(0), device=queries.device)[:, None], (1, 8)).view(-1)

        nb_mask = nb_ids > -1
        depth_feature = torch_scatter.scatter_sum(
            feature[nb_ids[nb_mask]] * nb_weight[nb_mask, None], pts_ids[nb_mask],
            dim=0, dim_size=queries.size(0)
        )

        if feature_bg is not None:
            non_nb_mask = nb_ids == -1
            depth_feature += torch_scatter.scatter_sum(
                feature_bg[None, :] * nb_weight[non_nb_mask, None], pts_ids[non_nb_mask],
                dim=0, dim_size=queries.size(0)
            )

        if compute_grad:
            nb_grad = nb_grad.view(-1, nb_grad.size(-1))
            depth_grad = torch_scatter.scatter_sum(
                feature[nb_ids[nb_mask]][:, :, None] * nb_grad[nb_mask, None, :],
                pts_ids[nb_mask], dim=0, dim_size=queries.size(0)
            )
            # Most of nb_grad[non_nb_mask] should be zero though...
            if feature_bg is not None:
                depth_grad += torch_scatter.scatter_sum(
                    feature_bg[None, :, None] * nb_grad[non_nb_mask, None, :],
                    pts_ids[non_nb_mask], dim=0, dim_size=queries.size(0)
                )
        else:
            depth_grad = None

        return depth_feature, depth_grad
