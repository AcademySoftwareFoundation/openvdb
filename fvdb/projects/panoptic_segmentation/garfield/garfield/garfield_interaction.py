"""Helper functions for interacting/visualization with GARField model."""
from typing import List, Optional, Tuple, Union
import viser
import trimesh
import torch.nn as nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

from garfield.garfield_model import GarfieldModel

class GarfieldClickScene(nn.Module):
    """UI for clicking on a scene (visualized as spheres).
    This needs to be a nn.Module to allow the viewer to register callbacks.
    """
    _click_handle: viser.GlbHandle
    _box_handle: viser.GlbHandle
    selected_location: np.ndarray
    scale_handle: ViewerSlider  # For getting the scale to query GARField
    model_handle: List[GarfieldModel]  # Store as list to avoid circular children

    def __init__(
            self,
            device: torch.device,
            scale_handle: ViewerSlider,
            model_handle: List[GarfieldModel]
        ):
        super().__init__()
        self.add_click_button: ViewerButton = ViewerButton(
            name="Click", cb_hook=self._add_click_cb
        )
        self.del_click_button: ViewerButton = ViewerButton(
            name="Reset Click", cb_hook=self._del_click_cb
        )
        self.viewer_control: ViewerControl = ViewerControl()

        self.scale_handle = scale_handle
        self.model_handle = model_handle
        self.scale_handle.cb_hook = self._update_scale_vis

        self._click_handle = None
        self._box_handle = None
        self.selected_location = None
        self.device = device

    def _add_click_cb(self, button: ViewerButton):
        """Button press registers a click event, which will add a sphere.
        Refer more to nerfstudio docs for more details. """
        self.add_click_button.set_disabled(True)
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.add_click_button.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Also keep track of the selected location."""

        origin = torch.tensor(click.origin).view(1, 3)
        direction = torch.tensor(click.direction).view(1, 3)

        # get intersection
        bundle = RayBundle(
            origin,
            direction,
            torch.tensor(0.001).view(1, 1),
            nears=torch.tensor(0.05).view(1, 1),
            fars=torch.tensor(100).view(1, 1),
            camera_indices=torch.tensor(0).view(1, 1),
        ).to(self.device)

        # Get the distance/depth to the intersection --> calculate 3D position of the click
        model = self.model_handle[0]
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
        if model.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        with torch.no_grad():
            depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        distance = depth[0, 0].detach().cpu().numpy()
        click_position = np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO

        # Update click visualization
        self._del_click_cb(None)
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
        sphere_mesh.vertices += click_position
        sphere_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/hit_pos", mesh=sphere_mesh
        )
        self._click_handle = sphere_mesh_handle
        self.selected_location = np.array(origin + direction * distance)
        self._update_scale_vis(self.scale_handle)

    def _del_click_cb(self, button: ViewerButton):
        """Remove the click location and click visualizations."""
        if self._click_handle is not None:
            self._click_handle.remove()
        self._click_handle = None
        if self._box_handle is not None:
            self._box_handle.remove()
        self._box_handle = None
        self.selected_location = None

    def _update_scale_vis(self, slider: ViewerSlider):
        """Update the scale visualization."""
        if self._box_handle is not None:
            self._box_handle.remove()
            self._box_handle = None
        if self.selected_location is not None:
            box_mesh = trimesh.creation.icosphere(radius=VISER_NERFSTUDIO_SCALE_RATIO*max(0.001, slider.value)/2, subdivision=0)
            self._box_handle = self.viewer_control.viser_server.add_mesh_simple(
                name=f"/hit_pos_box", 
                vertices=box_mesh.vertices,
                faces=box_mesh.faces,
                position=(self.selected_location * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )

    def get_outputs(self, outputs: dict):
        """Visualize affinity between the selected 3D point and the points visibl in current rendered view."""
        if self.selected_location is None:
            return None

        location = self.selected_location
        instance_scale = self.scale_handle.value
        
        # mimic the fields call
        grouping_field = self.model_handle[0].grouping_field
        positions = torch.tensor(location).view(1, 3).to(self.device)
        positions = grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        instance_pass = grouping_field.get_mlp(x, torch.tensor([instance_scale]).to(self.device).view(1, 1))

        return {
            "instance_interact": torch.norm(outputs['instance'] - instance_pass.float(), p=2, dim=-1)
        }