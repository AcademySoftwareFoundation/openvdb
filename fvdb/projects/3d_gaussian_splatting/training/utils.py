# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn.functional as F


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    @staticmethod
    def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = self._rotation_6d_to_matrix(drot + self.identity.expand(*batch_shape, -1))  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    @staticmethod
    def _eval_sh_bases_fast(basis_dim: int, dirs: torch.Tensor):
        """
        from gsplat:
            https://github.com/nerfstudio-project/gsplat/blob/ec3e715f5733df90d804843c7246e725582df10c/gsplat/cuda/_torch_impl.py#L620

        Evaluate spherical harmonics bases at unit direction for high orders
        using approach described by
        Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
        https://jcgt.org/published/0002/02/06/


        :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
        :param dirs: torch.Tensor (..., 3) unit directions

        :return: torch.Tensor (..., basis_dim)

        See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
        """
        result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)

        result[..., 0] = 0.2820947917738781

        if basis_dim <= 1:
            return result

        x, y, z = dirs.unbind(-1)

        fTmpA = -0.48860251190292
        result[..., 2] = -fTmpA * z
        result[..., 3] = fTmpA * x
        result[..., 1] = fTmpA * y

        if basis_dim <= 4:
            return result

        z2 = z * z
        fTmpB = -1.092548430592079 * z
        fTmpA = 0.5462742152960395
        fC1 = x * x - y * y
        fS1 = 2 * x * y
        result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
        result[..., 7] = fTmpB * x
        result[..., 5] = fTmpB * y
        result[..., 8] = fTmpA * fC1
        result[..., 4] = fTmpA * fS1

        if basis_dim <= 9:
            return result

        fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
        fTmpB = 1.445305721320277 * z
        fTmpA = -0.5900435899266435
        fC2 = x * fC1 - y * fS1
        fS2 = x * fS1 + y * fC1
        result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
        result[..., 13] = fTmpC * x
        result[..., 11] = fTmpC * y
        result[..., 14] = fTmpB * fC1
        result[..., 10] = fTmpB * fS1
        result[..., 15] = fTmpA * fC2
        result[..., 9] = fTmpA * fS2

        if basis_dim <= 16:
            return result

        fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
        fTmpC = 3.31161143515146 * z2 - 0.47308734787878
        fTmpB = -1.770130769779931 * z
        fTmpA = 0.6258357354491763
        fC3 = x * fC2 - y * fS2
        fS3 = x * fS2 + y * fC2
        result[..., 20] = 1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346) + -1.006230589874905 * (
            0.9461746957575601 * z2 - 0.3153915652525201
        )
        result[..., 21] = fTmpD * x
        result[..., 19] = fTmpD * y
        result[..., 22] = fTmpC * fC1
        result[..., 18] = fTmpC * fS1
        result[..., 23] = fTmpB * fC2
        result[..., 17] = fTmpB * fS2
        result[..., 24] = fTmpA * fC3
        result[..., 16] = fTmpA * fS3
        return result

    def forward(
        self, features: torch.Tensor, embed_ids: torch.Tensor, dirs: torch.Tensor, sh_degree: int
    ) -> torch.Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = self._eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors
