import torch
from typing import Callable, Union
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
        Abstract base class for SparseFeatureHierarchy.
    The full code should function normally if each function is correctly implemented.
    """

    @abstractmethod
    def __init__(self, depth: int, voxel_size: float, device, range_kernel: Callable[[int], torch.Tensor]):
        """
        Initialize the metadata of the hierarchy.
        :param depth: int, number of layers
        :param voxel_size: float, width of the voxel at the finest level.
        :param device: torch.Device, device where the data structure should reside
        :param range_kernel: a helper function that specifies the relative offsets in the kernel.
            *Note*: The sequence in the kernel only has to be respected when conv=True in get_self_neighbours!
        """
        pass

    @property
    @abstractmethod
    def depth(self) -> int:
        """
        :return: total depth of the tree
        """
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> float:
        """
        :return: width of the voxel at the finest level.
        """
        pass

    @abstractmethod
    def get_stride(self, depth: int) -> int:
        """
        :return: the stride at depth. Usually this would be 2**depth
        """
        pass

    @abstractmethod
    def get_coords(self, depth: int, expand: int = 0, conforming: bool = False) -> torch.Tensor:
        """
        :param depth:
        :param expand:
        :param conforming:
        :return: (N, 3) float32 torch.Tensor, each row is the bottom-left-near voxel corner coordinate in normalized space.
            Note: This might be called multiple times within a function, so we could possibly cache it
        instead of iterating over the tree multiple times.
        """
        pass

    @abstractmethod
    def get_num_voxels(self, depth: int) -> int:
        """
        :return: number of voxels in a given layer
        """
        pass

    @abstractmethod
    def get_voxel_centers(self, depth: int, normalized: bool = False):
        """
        Get the centroid coordinates of all existing voxels at depth.
        :param depth: int
        :param normalized: if True, then divide the coordinates by voxel size, so that unit 1 is a single voxel.
        :return: (N, 3) float32 torch.Tensor
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        :return: str
        """
        pass

    @abstractmethod
    def get_coords_neighbours(self, source_coords: torch.Tensor, source_stride: int, target_depth: int,
                              nn_kernel: torch.Tensor, conv_based: bool = False, transposed: bool = False, raw: bool = False):
        """
        Get neighbourhood information of source_coords.
        :param source_coords:
        :param source_stride:
        :param target_depth:
        :param nn_kernel:
        :param conv_based:
        :param transposed:
        :return:
        """
        pass

    @abstractmethod
    def get_self_neighbours(self, source_depth: int, target_depth: int, target_range: int,
                            conv_based: bool = False):
        """

        :param source_depth:
        :param target_depth:
        :param target_range:
        :param conv_based:
        :return:
        """
        pass

    @abstractmethod
    def evaluate_voxel_status(self, coords: torch.Tensor, depth: int):
        """
        Evaluate status in the hierarchy, please refer to core.hashtree.VoxelStatus for numerical values:
            VoxelStatus.VS_NON_EXIST: This voxel shouldn't exist
            VoxelStatus.VS_EXIST_STOP: This voxel exists and is a leaf node
            VoxelStatus.VS_EXIST_CONTINUE: This voxel exists and has >0 children
        :param coords: (N, 3) torch.Tensor coordinates in the world space
        :param depth: int
        :return: (N, ) long torch.Tensor, indicating voxel status
        """
        pass

    @abstractmethod
    def split_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor):
        """
        Obtain the tri-linearly interpolated data located at xyz.
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (M, K), where K is feature dimension, and M = self.get_num_voxels(data_depth)
        :return: (N, K) torch.Tensor
        """
        pass

    @abstractmethod
    def splat_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor = None,
                   check_corr: bool = True, return_nf_mask: bool = False):
        """
        Splat data located at xyz to the tree voxels.
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (N, K)
        :param check_corr: if True, check if data is fully supported by its 8 neighbours
        :param return_nf_mask: Legacy, do not use.
        :return: (M, K), where M = self.get_num_voxels(data_depth)
        """
        pass

    @abstractmethod
    def build_hierarchy_dense(self, xyz: torch.Tensor, expand_range: int = 0):
        """
        Ignore for now
        :param xyz:
        :param expand_range:
        :return:
        """
        pass

    @abstractmethod
    def build_hierarchy_subdivide(self, xyz: torch.Tensor, subdivide_policy, expand: bool = False,
                                  limit_adaptive_depth: int = 100, **policy_kwargs):
        """
        Ignore for now
        :param xyz:
        :param subdivide_policy:
        :param expand:
        :param limit_adaptive_depth:
        :param policy_kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_hierarchy_adaptive(self, xyz: torch.Tensor, xyz_density: torch.Tensor, log_base: float = 4.0,
                                 min_density: float = 8.0,
                                 limit_adaptive_depth: int = 100) -> torch.Tensor:
        """
        Build the hierarchy by first determine the integer level of each point (based on xyz_density, log_base and
        min_density), then splat the points onto the tree structure.
        :param xyz: (N, 3) torch.Tensor
        :param xyz_density: (N, ) float torch.Tensor
        :param log_base: float
        :param min_density: float, minimum density in each voxel. If exceed, go to coarser level.
        :param limit_adaptive_depth: int. Maximum adaptive number of levels.
        :return torch.Tensor long. (N, ) level that the point lies in.
        """
        pass

    @abstractmethod
    def update_coords(self, depth: int, coords: Union[torch.Tensor, None]):
        """
        Update the structure of the tree. This is mainly used during decoder's structure building stage.
            For now you could assume that the structure at depth does not exist yet.
            But I think we should have some general function that alters the tree structure.
        :param depth: int
        :param coords: torch.Tensor (N, 3) or None, if None, then this layer would be empty.
        :return:
            - new_coords: torch.Tensor (N, 3)
            - permutation: torch.Tensor (N, ):
                f[p] maps f from input-seq to fvdb-seq
                p[i] maps i from fvdb-seq to input-seq
        """
        pass

    @abstractmethod
    def trilinear_interpolate(self, queries: torch.Tensor, depth: int, feature: torch.Tensor,
                              feature_bg: torch.Tensor = None, compute_grad: bool = False):
        """
        Trilinearly interpolate the features, this is very similar to self.splat.
            Maybe merge them in the future.
        :param queries:
        :param depth:
        :param feature:
        :param feature_bg:
        :param compute_grad:
        :return:
        """
        pass

    def get_visualization(self, stylized: bool = False, render_level: bool = False):
        from pycg import vis

        tree_wireframes = []
        for d in range(self.depth):
            is_solid = stylized and d == 0
            d_min, d_max = 0, self.get_stride(d)
            if is_solid:
                d_min, d_max = 0.1 * self.get_stride(d), 0.9 * self.get_stride(d)
            if self.get_coords(d).size(0) == 0:
                continue
            blk_wireframe = vis.wireframe_bbox((self.get_coords(d) + d_min) * self.voxel_size,
                                               (self.get_coords(d) + d_max) * self.voxel_size,
                                               solid=is_solid, ucid=d if stylized else -1, tube=render_level,
                                               tube_radius=0.001)
            if render_level and d == 0:
                blk_wireframe = vis.transparent(blk_wireframe, alpha=0.5)
            tree_wireframes.append(blk_wireframe)

        return tree_wireframes
