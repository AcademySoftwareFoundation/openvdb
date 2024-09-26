# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
import unittest
from pathlib import Path

import imageio
import numpy as np
import OpenImageIO as oiio
import torch

from fvdb import JaggedTensor
from fvdb.utils import gaussian_fully_fused_projection, gaussian_render

from .common import get_fvdb_test_data_path


def compare_images(pixels_or_path_a, pixels_or_path_b):
    """Return true, if the two images perceptually differ

    Unlike what the documentation says here
    https://openimageio.readthedocs.io/en/master/imagebufalgo.html#_CPPv4N4OIIO12ImageBufAlgo11compare_YeeERK8ImageBufRK8ImageBufR14CompareResultsff3ROIi
    `compare_Yee` returns `False` if the images are the **same**.

    Populated entries of the `CompareResults` objects are `maxerror`, `maxx`, `maxy`, `maxz`, and `nfail`,
    """
    img_a = oiio.ImageBuf(pixels_or_path_a)
    img_b = oiio.ImageBuf(pixels_or_path_b)
    cmp = oiio.CompareResults()
    differ = oiio.ImageBufAlgo.compare_Yee(img_a, img_b, cmp)
    return differ, cmp


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class TestGaussianRender(unittest.TestCase):

    data_path = get_fvdb_test_data_path() / "gsplat"
    save_image_data = False
    # NB: The files for regression data are saved at pwd to prevent accidental overwrites
    save_regression_data = False

    def _load_test_data(self, data_path):
        data = np.load(data_path)
        self.means = torch.from_numpy(data["means3d"]).float().to(self.device)
        self.quats = torch.from_numpy(data["quats"]).float().to(self.device)
        self.scales = torch.from_numpy(data["scales"]).float().to(self.device)
        self.opacities = torch.from_numpy(data["opacities"]).float().to(self.device)
        self.colors = torch.from_numpy(data["colors"]).float().to(self.device)
        self.viewmats = torch.from_numpy(data["viewmats"]).float().to(self.device)
        self.Ks = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.width = data["width"].item()
        self.height = data["height"].item()

    def setUp(self):
        self.device = "cuda:0"

        self._load_test_data(self.data_path / "test_garden_cropped.npz")

        self.num_cameras = self.viewmats.shape[0]

        self.means.requires_grad = True
        self.quats.requires_grad = True
        self.scales.requires_grad = True
        self.opacities.requires_grad = True

        self.sh_degree = 3
        self.sh_coeffs = torch.zeros((self.means.shape[0], (self.sh_degree + 1) ** 2, 3), device=self.device)
        self.sh_coeffs[:, 0, :] = rgb_to_sh(self.colors)
        self.sh_coeffs.requires_grad = True

    def test_fully_fused_projection(self):
        radii, means2d, depths, conics = gaussian_fully_fused_projection(
            self.means, self.quats, self.scales, self.viewmats, self.Ks, self.width, self.height, 0.3, 0.01, 1e10, 0.0
        )

        if self.save_regression_data:
            torch.save(radii, "regression_radii.pt")
            torch.save(means2d, "regression_means2d.pt")
            torch.save(depths, "regression_depths.pt")
            torch.save(conics, "regression_conics.pt")

        # Regression test
        test_radii = torch.load(self.data_path / "regression_radii.pt", weights_only=True)
        test_means2d = torch.load(self.data_path / "regression_means2d.pt", weights_only=True)
        test_depths = torch.load(self.data_path / "regression_depths.pt", weights_only=True)
        test_conics = torch.load(self.data_path / "regression_conics.pt", weights_only=True)

        torch.testing.assert_close(radii, test_radii)
        torch.testing.assert_close(means2d[radii > 0], test_means2d[radii > 0])
        torch.testing.assert_close(depths[radii > 0], test_depths[radii > 0])
        torch.testing.assert_close(conics[radii > 0], test_conics[radii > 0])

    def _tensors_to_pixel(self, colors, alphas):
        canvas = (
            torch.cat(
                [
                    colors.reshape(self.num_cameras * self.height, self.width, 3),
                    alphas.reshape(self.num_cameras * self.height, self.width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return (canvas * 255).astype(np.uint8)

    def test_gaussian_render(self):

        # single scene rendering
        render_colors, render_alphas, means2d = gaussian_render(
            self.means.contiguous(),
            self.quats.contiguous(),
            self.scales.contiguous(),
            self.opacities.contiguous(),
            self.sh_coeffs.contiguous(),
            self.viewmats.contiguous(),
            self.Ks.contiguous(),
            self.width,
            self.height,
            0.3,
            0.01,
            1e10,
            0.0,
            self.sh_degree,
            16,
        )

        pixels = self._tensors_to_pixel(render_colors, render_alphas)
        differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_result.png"))

        if self.save_image_data:
            imageio.imsave(self.data_path / "output_gaussian_render.png", pixels)

        if self.save_regression_data:
            imageio.imsave("regression_gaussian_render_result.png", pixels)

        self.assertFalse(
            differ, f"Gaussian renders for Torch tensors differ from reference image at {cmp.nfail} pixels"
        )

    def test_gaussian_render_jagged(self):
        # There are two scenes
        jt_means = JaggedTensor([self.means, self.means]).to(self.device)
        jt_quats = JaggedTensor([self.quats, self.quats]).to(self.device)
        jt_scales = JaggedTensor([self.scales, self.scales]).to(self.device)
        jt_opacities = JaggedTensor([self.opacities, self.opacities]).to(self.device)
        jt_sh_coeffs = JaggedTensor([self.sh_coeffs, self.sh_coeffs]).to(self.device)

        # The first scene renders to 2 views and the second scene renders to a single view
        jt_viewmats = JaggedTensor([self.viewmats[:2], self.viewmats[2:]]).to(self.device)
        jt_Ks = JaggedTensor([self.Ks[:2], self.Ks[2:]]).to(self.device)

        # g_sizes = means.joffsets[1:] - means.joffsets[:-1]
        # c_sizes = Ks.joffsets[1:] - Ks.joffsets[:-1]
        # tt = g_sizes.repeat_interleave(c_sizes)
        # camera_ids = torch.arange(viewmats.rshape[0], device=device).repeat_interleave(tt, dim=0)

        # dd0 = means.joffsets[:-1].repeat_interleave(c_sizes, 0)
        # dd1 = means.joffsets[1:].repeat_interleave(c_sizes, 0)
        # shifts = dd0[1:] - dd1[:-1]
        # shifts = torch.cat([torch.tensor([0], device=device), shifts])  # [0, -1000, 0]
        # shifts_cumsum = shifts.cumsum(0)  # [0, -1000, -1000]
        # gaussian_ids = torch.arange(len(camera_ids), device=device)  # [0, 1, 2, ..., 2999]
        # gaussian_ids = gaussian_ids + shifts_cumsum.repeat_interleave(tt, dim=0)

        render_colors, render_alphas, means2d, camera_ids, gaussian_ids = gaussian_render(
            jt_means,
            jt_quats,
            jt_scales,
            jt_opacities,
            jt_sh_coeffs,
            jt_viewmats,
            jt_Ks,
            self.width,
            self.height,
            0.3,
            0.01,
            1e10,
            0.0,
            self.sh_degree,
            16,
        )
        torch.cuda.synchronize()

        pixels = self._tensors_to_pixel(render_colors, render_alphas)
        differ, cmp = compare_images(pixels, str(self.data_path / "regression_gaussian_render_jagged_result.png"))

        if self.save_image_data:
            imageio.imsave(self.data_path / "output_gaussian_render_jagged.png", pixels)

        if self.save_regression_data:
            imageio.imsave("regression_gaussian_render_jagged_result.png", pixels)

        self.assertFalse(
            differ, f"Gaussian renders for jagged tensors differ from reference image at {cmp.nfail} pixels"
        )


if __name__ == "__main__":
    test = TestGaussianRender()
    test.setUp()

    test.test_gaussian_render()
    test.test_gaussian_render_jagged()
    test.test_fully_fused_projection()
