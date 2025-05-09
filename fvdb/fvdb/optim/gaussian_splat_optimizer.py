# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import torch
import torch.nn.functional as F
import torch.optim

from .. import GaussianSplat3d


@dataclass
class GaussianSplatOptimizer:
    """Optimzier for training Gaussian Splat radiance fields over a collection of posed images.

    This optimizer uses Adam with a fixed learning rate for each parameter in a Gaussian Radiance field
    (i.e. means, covariances, opacities, spherical harmonics).
    It also handles splitting/duplicating/deleting Gaussians based on their opacity and gradients following the
    algorithm in the original Gaussian Splatting paper (https://arxiv.org/abs/2308.04079).

    Args:
        module: The module representing the Gaussian Splatting scene (e.g. fvdb.nn.GaussianSplat3D).
        mean_lr_decay_exponent: The exponent to decay the learning rate for the means. Default: 1.0.

        prune_opacity_threshold: The opacity threshold below which to prune a Gaussian. Default: 0.005.
        prune_scale3d_threshold: The 3D scale threshold above which to prune a Gaussian. Default: 0.1.
        prune_scale2d_threshold: The 2D scale threshold above which to prune a Gaussian. Default: 0.15.

        grow_grad2d_threshold: The 2D gradient threshold above which to grow a Gaussian. Default: 0.0002.
        grow_scale3d_threshold: The 3D scale threshold below which to grow a Gaussian. Default: 0.01.
        grow_scale2d_threshold: The 2D scale threshold below which to grow a Gaussian. Default: 0.05.

        absgrad: Whether to use the absolute value of the gradients for refinement. Default: False.
        revised_opacity: Whether to use the revised opacity formulation from https://arxiv.org/abs/2404.06109. Default: False.
    """

    module: GaussianSplat3d
    mean_lr_decay_exponent: float = 1.0
    scene_scale: float = 1.0

    prune_opacity_threshold: float = 0.005
    prune_scale3d_threshold: float = 0.1
    prune_scale2d_threshold: float = 0.15

    grow_grad2d_threshold: float = 0.0002
    grow_scale3d_threshold: float = 0.01
    grow_scale2d_threshold: float = 0.05

    absgrad: bool = False
    revised_opacity: bool = False

    def __post_init__(self):
        default_lrs = {
            "means": 1.6e-4 * self.scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }

        if self.absgrad:
            raise NotImplementedError("absgrad is not yet implemented")

        batch_size = 1

        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1 for more details.
        lr_batch_rescale = math.sqrt(float(batch_size))
        self._optimizers = {
            "means": torch.optim.Adam(
                [{"params": self.module.means, "lr": default_lrs["means"] * lr_batch_rescale, "name": "means"}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
            "scales": torch.optim.Adam(
                [{"params": self.module.log_scales, "lr": default_lrs["scales"] * lr_batch_rescale, "name": "scales"}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
            "quats": torch.optim.Adam(
                [{"params": self.module.quats, "lr": default_lrs["quats"] * lr_batch_rescale, "name": "quats"}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
            "opacities": torch.optim.Adam(
                [
                    {
                        "params": self.module.logit_opacities,
                        "lr": default_lrs["opacities"] * lr_batch_rescale,
                        "name": "opacities",
                    }
                ],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
            "sh0": torch.optim.Adam(
                [{"params": self.module.sh0, "lr": default_lrs["sh0"] * lr_batch_rescale, "name": "sh0"}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
            "shN": torch.optim.Adam(
                [{"params": self.module.shN, "lr": default_lrs["shN"] * lr_batch_rescale, "name": "shN"}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            ),
        }

        # means has a learning rate schedule, that end at 0.01 of the initial value
        self._means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizers["means"], gamma=self.mean_lr_decay_exponent
        )

        # Number of times we've called backward since zeroing the gradients
        self._num_grad_accumulation_steps = 1

        # Count the number of times we call backward between zeroing the gradients
        # If we're accumulating gradients, we need to know this to properly grow/split/prune
        def _count_accumulation_steps_backward_hook(_):
            self._num_grad_accumulation_steps += 1

        self.module.means.register_hook(_count_accumulation_steps_backward_hook)

    def step(self):
        for optimizer in self._optimizers.values():
            optimizer.step()
        self._means_lr_scheduler.step()

    def zero_grad(self, set_to_none: bool = False):
        self._num_grad_accumulation_steps = 0
        for optimizer in self._optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self._optimizers.items()},
            "means_lr_scheduler": self._means_lr_scheduler.state_dict(),
            "mean_lr_decay_exponent": self.mean_lr_decay_exponent,
            "prune_opacity_threshold": self.prune_opacity_threshold,
            "prune_scale3d_threshold": self.prune_scale3d_threshold,
            "prune_scale2d_threshold": self.prune_scale2d_threshold,
            "grow_grad2d_threshold": self.grow_grad2d_threshold,
            "grow_scale3d_threshold": self.grow_scale3d_threshold,
            "grow_scale2d_threshold": self.grow_scale2d_threshold,
            "absgrad": self.absgrad,
            "revised_opacity": self.revised_opacity,
            "version": 2,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict["version"] != 2:
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        for name, optimizer in self._optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        self._means_lr_scheduler.load_state_dict(state_dict["means_lr_scheduler"])
        self.mean_lr_decay_exponent = state_dict["mean_lr_decay_exponent"]
        self.prune_opacity_threshold = state_dict["prune_opacity_threshold"]
        self.prune_scale3d_threshold = state_dict["prune_scale3d_threshold"]
        self.prune_scale2d_threshold = state_dict["prune_scale2d_threshold"]
        self.grow_grad2d_threshold = state_dict["grow_grad2d_threshold"]
        self.grow_scale3d_threshold = state_dict["grow_scale3d_threshold"]
        self.grow_scale2d_threshold = state_dict["grow_scale2d_threshold"]
        self.absgrad = state_dict["absgrad"]
        self.revised_opacity = state_dict["revised_opacity"]

    @torch.no_grad()
    def refine_gaussians(self, use_scales: bool = False, use_screen_space_scales: bool = False):
        if use_screen_space_scales:
            if not self.module.track_max_2d_radii_for_grad:
                raise ValueError(
                    "use_screen_space_scales is set to True but the model is not configured to "
                    + "track screen space scales. Set model.track_max_2d_radii_for_grad = True."
                )
        # Grow the number of Gaussians via:
        # 1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
        # 2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
        #    or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold
        #
        # Note that splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
        # μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.
        n_dupli, n_split = self._grow_gs(use_screen_space_scales)
        # Prune Gaussians whose opacity is below a threshold or whose screen space spatial extent is too large
        n_prune = self._prune_gs(use_scales, use_screen_space_scales)
        # Reset running statistics used to determine which Gaussians to add/split/prune
        self.module.reset_grad_state()

        return n_dupli, n_split, n_prune

    @torch.no_grad()
    def reset_opacities(self):
        """Reset the opacities to the given (post-sigmoid) value."""

        value = self.prune_opacity_threshold * 2.0

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "opacities":
                return torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        new_opac = self._update_optimizer("opacities", param_fn, optimizer_fn)
        self.module.logit_opacities = new_opac

    @torch.no_grad()
    def _grow_gs(self, use_screen_space_scales) -> Tuple[int, int]:
        """
        Grow the number of Gaussians via:
          1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
          2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
             or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold

        Note: Splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
              μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.

        Args:
            use_screen_space_scales: If set to true, use the tracked screen space scales to decide whether to split.
                                     Note that the model must have been configured to track these scales by setting
                                     GaussianSplat3d.track_max_2d_radii_for_grad = True.
        """

        # We use the average gradient ( over the the last N steps) of the projected Gaussians with respect to the
        # loss to decide which Gaussians to add/split/prune
        # count is the number of times a Gaussian has been projected (i.e. included in the loss gradient computation)
        # grad_2d is the sum of the gradients of the projected Gaussians (dL/dμ2D) over the last N steps
        count = self.module.accumulated_gradient_step_counts_for_grad.clamp_min(1)
        if self._num_grad_accumulation_steps > 1:
            count *= self._num_grad_accumulation_steps

        grads = self.module.accumulated_mean_2d_gradient_norms_for_grad / count
        device = grads.device

        # If the 2D projected gradient is high and the spatial size is small, duplicate the Gaussian
        is_grad_high = grads > self.grow_grad2d_threshold
        is_small = self.module.scales.max(dim=-1).values <= self.grow_scale3d_threshold * self.scene_scale
        is_dupli = is_grad_high & is_small
        n_dupli: int = int(is_dupli.sum().item())

        # If the 2D projected gradient is high and the spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large

        # If the 2D projected spatial extent exceeds the threshold, split the Gaussian
        if use_screen_space_scales:
            is_split |= self.module.accumulated_max_2d_radii_for_grad > self.grow_scale2d_threshold
        n_split: int = int(is_split.sum().item())

        # Hardcode these for now but could be made configurable
        dup_factor = 1  # 1 means one gaussian becomes 2, 2 means one gaussian becomes 3, etc.
        split_factor = 2

        # First duplicate the Gaussians
        if n_dupli > 0:
            self.duplicate_gaussians(mask=is_dupli, dup_factor=dup_factor)

        # Track new Gaussians added by duplication so we we don't split them
        is_split = torch.cat([is_split] + [torch.zeros(n_dupli, dtype=torch.bool, device=device)] * dup_factor)

        # Now split the Gaussians
        if n_split > 0:
            self.subdivide_gaussians(mask=is_split, split_factor=split_factor)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, use_scales: bool = False, use_screen_space_scales: bool = False) -> int:
        # Prune any Gaussians whose opacity is below the threshold or whose (2D projected) spatial extent is too large
        is_prune = self.module.opacities.flatten() < self.prune_opacity_threshold
        if use_scales:
            is_too_big = self.module.scales.max(dim=-1).values > self.prune_scale3d_threshold * self.scene_scale
            # The INRIA code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but it doesn't really get used
            if use_screen_space_scales:
                is_too_big |= self.module.accumulated_max_2d_radii_for_grad > self.prune_scale2d_threshold

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            self.remove_gaussians(mask=is_prune)

        return int(n_prune)

    @torch.no_grad()
    def subdivide_gaussians(self, mask: torch.Tensor, split_factor: int = 2):
        """Split the Gaussian with the given mask.

        Args:
            mask: A boolean mask with shape [num_means,] indicating which Gaussians to split.
            split_factor: The number of splits for each Gaussian. Default: 4.
        """

        def _normalized_quat_to_rotmat(quat_: torch.Tensor) -> torch.Tensor:
            """Convert normalized quaternion to rotation matrix.

            Args:
                quat: Normalized quaternion in wxyz convension. (..., 4)

            Returns:
                Rotation matrix (..., 3, 3)
            """
            assert quat_.shape[-1] == 4, quat_.shape
            w, x, y, z = torch.unbind(quat_, dim=-1)
            mat = torch.stack(
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                    2 * (x * y + w * z),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - w * x),
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x**2 + y**2),
                ],
                dim=-1,
            )
            return mat.reshape(quat_.shape[:-1] + (3, 3))

        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = self.module.scales[sel]  # [N,]
        quats = F.normalize(self.module.quats[sel], dim=-1)
        rotmats = _normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(split_factor, len(scales), 3, device=device),
        )  # [S, N, 3]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            repeats = [split_factor] + [1] * (p.dim() - 1)
            cat_dim = 0
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)  # [S*N, 3]
                p_rest = p[rest]
            elif name == "scales":
                # TODO: Adjust scale factor for splitting
                p_split = torch.log(scales / 1.6).repeat(split_factor, 1)  # [2N, 3]
                p_rest = p[rest]
            elif name == "opacities" and self.revised_opacity:
                new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
                p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
                p_rest = p[rest]
            elif name == "sh0" or name == "shN":
                repeats = [1] + [split_factor] + [1] * (p.dim() - 2)
                p_split = p[:, sel, ...].repeat(repeats)  # [K, 2N, D]
                p_rest = p[:, rest, ...]
                cat_dim = 1
            else:
                p_split = p[sel].repeat(repeats)
                p_rest = p[rest]
            p_new = torch.cat([p_rest, p_split], dim=cat_dim)
            return p_new

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                v_split = torch.zeros((v.shape[0], split_factor * len(sel), *v.shape[2:]), device=device)
                v_rest = v[:, rest, ...]
                cat_dim = 1
            else:
                v_split = torch.zeros((split_factor * len(sel), *v.shape[1:]), device=device)
                v_rest = v[rest]
                cat_dim = 0
            return torch.cat([v_rest, v_split], dim=cat_dim)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def duplicate_gaussians(self, mask: torch.Tensor, dup_factor: int = 1):
        """Duplicate the Gaussian with the given mask.

        Args:
            mask: A boolean mask of shape [num_means,] indicating which Gaussians to duplicate.
            dup_factor: The number of times to duplicate the selected Gaussians.
        """
        device = mask.device
        sel = torch.where(mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            cat_dim = 0
            repeats = [dup_factor] + [1] * (p.dim() - 1)
            if name == "sh0" or name == "shN":
                repeats = [1, dup_factor, 1]
                cat_dim = 1
                p_sel = p[:, sel, ...]
            else:
                p_sel = p[sel]
            return torch.cat([p, p_sel.repeat(repeats)], dim=cat_dim)

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                zpad = torch.zeros(v.shape[0], len(sel) * dup_factor, *v.shape[2:], device=v.device, dtype=v.dtype)
                return torch.cat([v, zpad], dim=1)
            else:
                zpad = torch.zeros((len(sel) * dup_factor, *v.shape[1:]), device=device)
                return torch.cat([v, zpad])

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def remove_gaussians(self, mask: torch.Tensor):
        """Remove the Gaussian with the given mask.

        Args:
            mask: A boolean mask of shape [num_means,] indicating which Gaussians to remove.
        """
        sel = torch.where(~mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                return p[:, sel, ...]
            return p[sel]

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                return v[:, sel, ...]
            else:
                return v[sel]

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)

    @torch.no_grad()
    def _update_optimizer(
        self,
        name: str,
        param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        optimizer = self._optimizers[name]
        ret = None
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    p_state[key] = optimizer_fn(name, key, v)
            p_new = param_fn(name, p)
            p_new.requires_grad = True
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            ret = p_new
        assert ret is not None
        return ret

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
        names: Union[List[str], None] = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            names: A list of key names to update. If None, update all. Default: None.
        """
        params = {
            "means": self.module.means,
            "scales": self.module.log_scales,
            "quats": self.module.quats,
            "opacities": self.module.logit_opacities,
            "sh0": self.module.sh0,
            "shN": self.module.shN,
        }
        if names is None:
            # If names is not provided, update all parameters
            names = list(params.keys())

        for name in names:
            params[name] = self._update_optimizer(name, param_fn, optimizer_fn)
        self.module.set_state(
            params["means"],
            params["quats"],
            params["scales"],
            params["opacities"],
            params["sh0"],
            params["shN"],
            True,
        )
