# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim

from ..nn import GaussianSplat3D


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

        refine_start_iter: The iteration to start refining (prune/split/dup) the Gausians. Default: 500.
        refine_every: The number of iterations between each refinement. Default: 100.
        refine_stop_iter: The iteration to stop refining (prune/split/dup) the Gausians. Default: 15_000.
        reset_every: The number of iterations between resetting the opacity parameters. Default: 3000.

        refine_scale2d_stop_iter: The iteration to stop using the screen space size of the Gaussians for refinement (prune/split/dup). Default: 0.
        pause_refine_after_reset: The number of iterations to pause refining (prune/split/dup) after resetting the opacity parameters. Default: 0.

        absgrad: Whether to use the absolute value of the gradients for refinement. Default: False.
        revised_opacity: Whether to use the revised opacity formulation from https://arxiv.org/abs/2404.06109. Default: False.

        verbose: Whether to print the number of Gaussians added/split/pruned at each step. Default: True.
    """

    module: GaussianSplat3D
    mean_lr_decay_exponent: float = 1.0

    prune_opacity_threshold: float = 0.005
    prune_scale3d_threshold: float = 0.1
    prune_scale2d_threshold: float = 0.15

    grow_grad2d_threshold: float = 0.0002
    grow_scale3d_threshold: float = 0.01
    grow_scale2d_threshold: float = 0.05

    refine_start_iter: int = 500
    refine_every: int = 100
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_scale2d_stop_iter: int = 0
    pause_refine_after_reset: int = 0

    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = True
    _key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def __post_init__(self):
        default_lrs = {
            "means": 1.6e-4 * self.module.scene_size,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }

        batch_size = 1

        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1 for more details.
        lr_batch_rescale = math.sqrt(float(batch_size))
        self._optimizers = {
            name: torch.optim.Adam(
                [{"params": self.module._params[name], "lr": default_lrs[name] * lr_batch_rescale, "name": name}],
                eps=1e-15 / lr_batch_rescale,
                betas=(1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999)),
            )
            for name, _ in default_lrs.items()
        }

        # means has a learning rate schedule, that end at 0.01 of the initial value
        self._means_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizers["means"], gamma=self.mean_lr_decay_exponent
        )

        self._state = {"grad2d": None, "count": None, "scene_scale": self.module.scene_size}
        if self.refine_scale2d_stop_iter > 0:
            self._state["radii"] = None

        # Number of times we've called backward since zeroing the gradients
        self._num_grad_accumulation_steps = 0

        # Retain the gradients of the 2D means of the Gaussians on the forward pass
        # We need it to decide which Gaussians to add/split/prune
        def _retain_grad_forward_hook(module, mod_in, mod_out):
            info_ = mod_out[-1]
            assert self._key_for_gradient in info_, "The 2D means of the Gaussians is required but missing."
            grad_param = info_[self._key_for_gradient]
            if grad_param.requires_grad:
                grad_param.retain_grad()

        # Count the number of times we call backward between zeroing the gradients
        # If we're accumulating gradients, we need to know this to properly grow/split/prune
        def _count_accumulation_steps_backward_hook(module, grad_in, grad_out):
            self._num_grad_accumulation_steps += 1

        self.module.register_forward_hook(_retain_grad_forward_hook)
        self.module.register_full_backward_hook(_count_accumulation_steps_backward_hook)

        self._step_count = 0

        self._logger = logging.getLogger(__name__ + ".GaussianSplatOptimizer")

    def step(self, info: Dict[str, Any]):
        self._step_internal(self._step_count, info)
        for optimizer in self._optimizers.values():
            optimizer.step()
        self._means_lr_scheduler.step()
        self._step_count += 1
        self.module.clear_cache()

    def zero_grad(self, set_to_none: bool = False):
        self._num_grad_accumulation_steps = 0
        for optimizer in self._optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self._optimizers.items()},
            "means_lr_scheduler": self._means_lr_scheduler.state_dict(),
            "state": self._state,
            "step_count": self._step_count,
            "mean_lr_decay_exponent": self.mean_lr_decay_exponent,
            "prune_opacity_threshold": self.prune_opacity_threshold,
            "prune_scale3d_threshold": self.prune_scale3d_threshold,
            "prune_scale2d_threshold": self.prune_scale2d_threshold,
            "grow_grad2d_threshold": self.grow_grad2d_threshold,
            "grow_scale3d_threshold": self.grow_scale3d_threshold,
            "grow_scale2d_threshold": self.grow_scale2d_threshold,
            "refine_start_iter": self.refine_start_iter,
            "refine_every": self.refine_every,
            "refine_stop_iter": self.refine_stop_iter,
            "reset_every": self.reset_every,
            "refine_scale2d_stop_iter": self.refine_scale2d_stop_iter,
            "pause_refine_after_reset": self.pause_refine_after_reset,
            "absgrad": self.absgrad,
            "revised_opacity": self.revised_opacity,
            "verbose": self.verbose,
            "key_for_gradient": self._key_for_gradient,
            "version": 1,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict["version"] != 1:
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        for name, optimizer in self._optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        self._means_lr_scheduler.load_state_dict(state_dict["means_lr_scheduler"])
        self._state = state_dict["state"]
        self._step_count = state_dict["step_count"]
        self.mean_lr_decay_exponent = state_dict["mean_lr_decay_exponent"]
        self.prune_opacity_threshold = state_dict["prune_opacity_threshold"]
        self.prune_scale3d_threshold = state_dict["prune_scale3d_threshold"]
        self.prune_scale2d_threshold = state_dict["prune_scale2d_threshold"]
        self.grow_grad2d_threshold = state_dict["grow_grad2d_threshold"]
        self.grow_scale3d_threshold = state_dict["grow_scale3d_threshold"]
        self.grow_scale2d_threshold = state_dict["grow_scale2d_threshold"]
        self.refine_start_iter = state_dict["refine_start_iter"]
        self.refine_every = state_dict["refine_every"]
        self.refine_stop_iter = state_dict["refine_stop_iter"]
        self.reset_every = state_dict["reset_every"]
        self.refine_scale2d_stop_iter = state_dict["refine_scale2d_stop_iter"]
        self.pause_refine_after_reset = state_dict["pause_refine_after_reset"]
        self.absgrad = state_dict["absgrad"]
        self.revised_opacity = state_dict["revised_opacity"]
        self.verbose = state_dict["verbose"]
        self._key_for_gradient = state_dict["key_for_gradient"]

    def _step_internal(self, step: int, info: Dict[str, Any]):
        """
        This method gets executed every time you call optimizer.step().

        It updates the internal state which is used to periodically, grow, split, and
        prune the Gaussians being optimized, and performs the actual grow/split/prune
        when the right number of steps are reached.

        See the comments for _update_state and _grow_gs, _prune_gs, and _split_gs for
        more details on these individual procedures.
        """
        if step >= self.refine_stop_iter:
            return

        # Update running statistics used to decide which Gaussians are to be added/split/pruned
        self._update_state(info)

        # If the current step is a multiple of refine_every, grow/split/prune the Gaussians
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # Grow the number of Gaussians via:
            # 1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
            # 2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
            #    or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold
            #
            # Note that splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
            # μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.
            n_dupli, n_split = self._grow_gs(step)
            if self.verbose:
                self._logger.info(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(self.module._params['means'])} GSs."
                )

            # Prune Gaussians whose opacity is below a threshold or whose screen space spatial extent is too large
            n_prune = self._prune_gs(step)

            # Log what we've done if the user wants to log
            if self.verbose:
                self._logger.info(
                    f"Step {step}: {n_prune} GSs pruned. " f"Now having {len(self.module._params['means'])} GSs."
                )

            # Reset running statistics used to determine which Gaussians to add/split/prune
            self._state["grad2d"].zero_()
            self._state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                self._state["radii"].zero_()
            torch.cuda.empty_cache()  # Save some memory but kind of slow

        # Reset all the opacity parameters to 2x the pruning threshold
        # This lets the optimizer explore the space more effectively
        if step % self.reset_every == 0:
            self._reset_opacity_parameters(self.prune_opacity_threshold * 2.0)

    def _update_state(self, info: Dict[str, Any]):
        packed: bool = False
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self._key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self._key_for_gradient].absgrad.clone()
        else:
            grads = info[self._key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(self.module._params.values())[0])

        if self._state["grad2d"] is None:
            self._state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if self._state["count"] is None:
            self._state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and self._state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            self._state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        self._state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        self._state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            self._state["radii"][gs_ids] = torch.maximum(
                self._state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(self, step: int) -> Tuple[int, int]:
        """
        Grow the number of Gaussians via:
          1. Duplicating those whose loss gradients are high and spatial size are small (i.e. have small eigenvals)
          2. Splitting those whose loss gradients are high and spatial size are large (i.e. have large eigenvals)
             or whose (2D projected) spatial extent simply exceeds the threshold self.grow_scale2d_threshold

        Note: Splitting a Gaussian with mean μ and covariance Σ is implemented by sampling two new means
              μ1, μ2 from N(μ, Σ), and setting the covariances Σ1 and Σ2 by dividing the eigenvalues of Σ by 1.6.

        Args:
            step: The current optimization step.
        """
        params = self.module._params

        # We use the average gradient ( over the the last N steps) of the projected Gaussians with respect to the
        # loss to decide which Gaussians to add/split/prune
        # count is the number of times a Gaussian has been projected (i.e. included in the loss gradient computation)
        # grad_2d is the sum of the gradients of the projected Gaussians (dL/dμ2D) over the last N steps
        count = self._state["count"] * self._num_grad_accumulation_steps
        grads = self._state["grad2d"] / count.clamp_min(1)
        device = grads.device

        # If the 2D projected gradient is high and the spatial size is small, duplicate the Gaussian
        is_grad_high = grads > self.grow_grad2d_threshold
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d_threshold * self._state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        # If the 2D projected gradient is high and the spatial size is large, split the Gaussian
        is_large = ~is_small
        is_split = is_grad_high & is_large

        # If the 2D projected spatial extent exceeds the threshold, split the Gaussian
        if step < self.refine_scale2d_stop_iter:
            is_split |= self._state["radii"] > self.grow_scale2d_threshold
        n_split = is_split.sum().item()

        # Hardcode these for now but could be made configurable
        dup_factor = 1
        split_factor = 2

        # First duplicate the Gaussians
        if n_dupli > 0:
            self._duplicate_params(mask=is_dupli, dup_factor=dup_factor)

        # Track new Gaussians added by duplication so we we don't split them
        is_split = torch.cat([is_split] + [torch.zeros(n_dupli, dtype=torch.bool, device=device)] * dup_factor)

        # Now split the Gaussians
        if n_split > 0:
            self._split_params(mask=is_split, revised_opacity=self.revised_opacity, split_factor=split_factor)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, step: int) -> int:
        params = self.module._params

        # Prune any Gaussians whose opacity is below the threshold or whose (2D projected) spatial extent is too large
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opacity_threshold
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d_threshold * self._state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= self._state["radii"] > self.prune_scale2d_threshold

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            self._remove_params(mask=is_prune)

        return int(n_prune)

    @torch.no_grad()
    def _split_params(self, mask: torch.Tensor, revised_opacity: bool = False, split_factor: int = 2):
        """Split the Gaussian with the given mask.

        Args:
            mask: A boolean mask with shape [num_means,] indicating which Gaussians to split.
            revised_opacity: Whether to use revised opacity formulation
                             from https://arxiv.org/abs/2404.06109. Default: False.
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

        scales = torch.exp(self.module._params["scales"][sel])  # [N,]
        quats = F.normalize(self.module._params["quats"][sel], dim=-1)
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
            elif name == "opacities" and revised_opacity:
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
            p_new = torch.nn.Parameter(torch.cat([p_rest, p_split], dim=cat_dim))
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
        # update the extra running state
        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                repeats = [split_factor] + [1] * (v.dim() - 1)
                v_new = v[sel].repeat(repeats)
                self._state[k] = torch.cat((v[rest], v_new))

    @torch.no_grad()
    def _duplicate_params(self, mask: torch.Tensor, dup_factor: int = 1):
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

            return torch.nn.Parameter(torch.cat([p, p_sel.repeat(repeats)], dim=cat_dim))

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                zpad = torch.zeros(v.shape[0], len(sel) * dup_factor, *v.shape[2:], device=v.device, dtype=v.dtype)
                return torch.cat([v, zpad], dim=1)
            else:
                zpad = torch.zeros((len(sel) * dup_factor, *v.shape[1:]), device=device)
                return torch.cat([v, zpad])

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)
        # update the extra running state
        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                vsel = v[sel]
                self._state[k] = torch.cat([v] + [vsel] * dup_factor)

    @torch.no_grad()
    def _remove_params(self, mask: torch.Tensor):
        """Remove the Gaussian with the given mask.

        Args:
            mask: A boolean mask of shape [num_means,] indicating which Gaussians to remove.
        """
        sel = torch.where(~mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                return torch.nn.Parameter(p[:, sel, ...])
            return torch.nn.Parameter(p[sel])

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name == "sh0" or name == "shN":
                return v[:, sel, ...]
            else:
                return v[sel]

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn)
        # update the extra running state
        for k, v in self._state.items():
            if isinstance(v, torch.Tensor):
                self._state[k] = v[sel]

    @torch.no_grad()
    def _reset_opacity_parameters(self, value: float):
        """Reset the opacities to the given (post-sigmoid) value.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            value: The value to reset the opacities
        """

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name == "opacities":
                opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
                return torch.nn.Parameter(opacities)
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        def optimizer_fn(name: str, key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(param_fn, optimizer_fn, names=["opacities"])

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
        params = self.module._params
        optimizers = self._optimizers

        if names is None:
            # If names is not provided, update all parameters
            names = list(params.keys())

        for name in names:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = optimizer_fn(name, key, v)
                p_new = param_fn(name, p)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                params[name] = p_new
