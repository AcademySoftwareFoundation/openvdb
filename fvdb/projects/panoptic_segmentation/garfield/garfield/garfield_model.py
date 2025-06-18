from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from garfield.garfield_field import GarfieldField, GarfieldFieldConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer.viewer_elements import *
from torch.nn import Parameter
from torchtyping import TensorType


class FeatureRenderer(nn.Module):
    """Render feature embeddings along  a ray, where features are unit norm"""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True)
        return output


@dataclass
class GarfieldModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: GarfieldModel)
    instance_field: GarfieldFieldConfig = field(default_factory=lambda: GarfieldFieldConfig())

    max_grouping_scale: float = 2.0
    """Maximum scale to use for grouping supervision. Should be set during pipeline init."""

    num_feat_samples: int = 24
    """Number of samples per ray to use for grouping supervision."""

    use_hierarchy_losses: bool = True
    use_single_scale: bool = False
    """For ablation only. For full GARField, keep hierarchy=True and single_scale=False."""


class GarfieldModel(NerfactoModel):
    config: GarfieldModelConfig
    grouping_field: GarfieldField

    def populate_modules(self):
        super().populate_modules()
        self.renderer_feat = FeatureRenderer()
        self.config.instance_field.use_single_scale = self.config.use_single_scale
        self.grouping_field = self.config.instance_field.setup()

        # Add a slider to the viewer to control the scale of the grouping field.
        self.scale_slider = ViewerSlider("Scale", 0.0, 0.0, 2.0, 0.001)

        # Store reference to click interface for GARField.
        # Note the List[GarfieldModel] is to avoid circular children.
        from garfield.garfield_interaction import GarfieldClickScene

        self.click_scene: GarfieldClickScene = GarfieldClickScene(
            device=("cuda" if torch.cuda.is_available() else "cpu"), scale_handle=self.scale_slider, model_handle=[self]
        )

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, TensorType]:
        outputs = super().get_outputs(ray_bundle)

        if self.grouping_field.quantile_transformer is None:
            # If scale statistics are not available, it's not possible to calculate grouping features.
            return outputs

        # Recalculate ray samples and weights
        # ... only if the model is in eval mode, where it should be no_grad().
        # If in training mode, `outputs` should already have calculated ray samples and weights.
        # Without this if-block, camera optimizer? gradients? seem to get messed up.
        ray_samples: RaySamples
        if self.training:
            ray_samples, weights = outputs["ray_samples_list"][-1], outputs["weights_list"][-1]
        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=self.density_fns
            )
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        # Choose the top k samples with the highest weights, to be used for grouping.
        # This is to decrease # of samples queried for grouping, while sampling close to the scene density.
        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        grouping_weights, best_ids = torch.topk(weights, self.config.num_feat_samples, dim=-2, sorted=False)
        grouping_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        # Define the scale for each sample. If the scale is not provided, use the selected scale.
        # "scale" is included in ray_bundle.metadata only from training batches, but
        # this would be good way to override the scale during inference.
        if self.training and ("scale" in ray_bundle.metadata):
            scales = ray_bundle.metadata["scale"]
            instance_scales = scales.view(grouping_samples.shape[0], 1)
        elif "scale" in ray_bundle.metadata:
            scales = ray_bundle.metadata["scale"]
            instance_scales = scales.view(grouping_samples.shape[0], 1)
        else:
            slider_value = self.scale_slider.value
            instance_scales = torch.ones(grouping_samples.shape[0], 1, device=self.device) * slider_value

        # Calculate features for the scale-conditioned grouping field.
        # Hash values need to be included in the outputs for the loss calculation.
        hash = self.grouping_field.get_hash(grouping_samples)
        hash_rendered = self.renderer_feat(embeds=hash, weights=grouping_weights.detach().half())
        if self.training:
            outputs["instance_hash"] = hash_rendered  # normalized!
        outputs["instance"] = self.grouping_field.get_mlp(hash_rendered, instance_scales).float()

        # If a click point is available, calculate the affinity between the click point and the scene.
        click_output = self.click_scene.get_outputs(outputs)
        if click_output is not None:
            outputs.update(click_output)

        return outputs

    @torch.no_grad()
    def get_grouping_at_points(self, positions: TensorType, scale: float) -> TensorType:
        """Get the grouping features at a set of points, given a scale."""
        # Apply distortion, calculate hash values, then normalize
        positions = self.grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in self.grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)

        # Calculate grouping features; create a scale tensor to match the batch size
        instance_scale = torch.ones((x.shape[0], 1), device=self.device) * scale
        return self.grouping_field.get_mlp(x, instance_scale)

    def get_loss_dict_group(self, outputs, batch, metrics_dict=None):
        # loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training:
            return

        loss_dict = {}
        margin = 1.0

        ####################################################################################
        # Calculate GT labels for the positive and negative pairs
        ####################################################################################
        # TODO(cmk) want to make this a little more efficient and cleaner
        input_id1 = input_id2 = batch["mask_id"]

        # Expand labels
        labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
        labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)

        # Mask for positive/negative pairs across the entire matrix
        mask_full_positive = labels1_expanded == labels2_expanded
        mask_full_negative = ~mask_full_positive

        # Create a block mask to only consider pairs within the same image -- no cross-image pairs
        chunk_size = batch["nPxImg"]  # i.e., the number of rays per image
        num_chunks = input_id1.shape[0] // chunk_size  # i.e., # of images in the batch
        block_mask = torch.kron(
            torch.eye(num_chunks, device=self.device, dtype=bool),
            torch.ones((chunk_size, chunk_size), device=self.device, dtype=bool),
        )  # block-diagonal matrix, to consider only pairs within the same image

        # Only consider upper triangle to avoid double-counting
        block_mask = torch.triu(block_mask, diagonal=0)
        # Only consider pairs where both points are valid (-1 means not in mask / invalid)
        block_mask = block_mask * (labels1_expanded != -1) * (labels2_expanded != -1)

        # Mask for diagonal elements (i.e., pairs of the same point).
        # Don't consider these pairs for grouping supervision (pulling), since they are trivially similar.
        diag_mask = torch.eye(block_mask.shape[0], device=self.device, dtype=bool)

        hash_rendered = outputs["instance_hash"]
        scale = batch["scale"].view(-1, 1)

        ####################################################################################
        # Grouping supervision
        ####################################################################################
        total_loss = 0

        # 1. If (A, s_A) and (A', s_A) in same group, then supervise the features to be similar
        # Note that `use_single_scale` (for ablation only) causes grouping_field to ignore the scale input.
        instance = self.grouping_field.get_mlp(hash_rendered, scale)
        mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
        instance_loss_1 = torch.norm(instance[mask[0]] - instance[mask[1]], p=2, dim=-1).nansum()
        total_loss += instance_loss_1

        # 2. If ", then also supervise them to be similar at s > s_A
        if self.config.use_hierarchy_losses and (not self.config.use_single_scale):
            scale_diff = torch.max(torch.zeros_like(scale), (self.config.max_grouping_scale - scale))
            larger_scale = scale + scale_diff * torch.rand(size=(1,), device=scale.device)
            instance = self.grouping_field.get_mlp(hash_rendered, larger_scale)
            mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
            instance_loss_2 = torch.norm(instance[mask[0]] - instance[mask[1]], p=2, dim=-1).nansum()
            total_loss += instance_loss_2

        # 4. Also supervising A, B to be dissimilar at scales s_A, s_B respectively seems to help.
        instance = self.grouping_field.get_mlp(hash_rendered, scale)
        mask = torch.where(mask_full_negative * block_mask)
        instance_loss_4 = (F.relu(margin - torch.norm(instance[mask[0]] - instance[mask[1]], p=2, dim=-1))).nansum()
        total_loss += instance_loss_4

        loss_dict["instance_loss"] = total_loss / torch.sum(block_mask).float()

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["garfield"] = list(self.grouping_field.parameters())
        return param_groups

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.
        This is the same as the base model's, but with a try/except in the case the shape is incorrect.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            try:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            except:
                pass
        return outputs
