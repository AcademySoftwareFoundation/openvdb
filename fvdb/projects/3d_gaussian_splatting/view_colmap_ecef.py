# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import tyro
import viser
import viser.transforms as tf

# TODO update imports with new api
from datasets._colmap_utils.rotation import Quaternion
from datasets.colmap_dataset import (
    ColmapParser,
    align_principle_axes,
    geo_norm_ecef2enu,
    transform_cameras,
    transform_points,
)
from tqdm.auto import tqdm


def main(
    colmap_path: Path = Path(__file__).parent / "assets/colmap_garden/sparse/0",
    images_path: Path = Path(__file__).parent / "assets/colmap_garden/images_8",
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    colmap_path = Path("/workspace/Data/bah/march_25_drone_data/glomap_run")
    downsample_factor = 20

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """For each client that connects, create GUI elements for adjusting the
        near/far clipping planes."""

        client.camera.far = 20000

        near_slider = client.gui.add_slider("Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near)
        far_slider = client.gui.add_slider("Far", min=1, max=20000.0, step=0.1, initial_value=client.camera.far)

        @near_slider.on_update
        def _(_) -> None:
            client.camera.near = near_slider.value

        @far_slider.on_update
        def _(_) -> None:
            client.camera.far = far_slider.value

    # Load the colmap info.
    # cameras = read_cameras_binary(colmap_path / "cameras.bin")
    # images = read_images_binary(colmap_path / "images.bin")
    # points3d = read_points3d_binary(colmap_path / "points3D.bin")
    parser = ColmapParser(str(colmap_path), normalization_type="ecef2enu")
    points = parser.points

    server.gui.add_text(
        "pnt mins (xyz)",
        ",".join(
            [
                "{:.2f}".format(np.min(points[:, 0])),
                "{:.2f}".format(np.min(points[:, 1])),
                "{:.2f}".format(np.min(points[:, 2])),
            ]
        ),
    )

    server.gui.add_text(
        "pnt maxs (xyz)",
        ",".join(
            [
                "{:.2f}".format(np.max(points[:, 0])),
                "{:.2f}".format(np.max(points[:, 1])),
                "{:.2f}".format(np.max(points[:, 2])),
            ]
        ),
    )

    server.gui.add_text(
        "pnt extent (xyz)",
        ",".join(
            [
                "{:.2f}".format(np.max(points[:, 0]) - np.min(points[:, 0])),
                "{:.2f}".format(np.max(points[:, 1]) - np.min(points[:, 1])),
                "{:.2f}".format(np.max(points[:, 2]) - np.min(points[:, 2])),
            ]
        ),
    )

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    npnts = points.shape[0]
    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=npnts,
        step=1,
        initial_value=int(npnts / 2),
    )

    nimages = len(parser.image_names)
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=nimages,
        step=1,
        initial_value=nimages,
    )
    gui_point_size = server.gui.add_slider("Point size", min=0.01, max=100, step=0.001, initial_value=0.05)

    # points = np.array([points3d[p_id].xyz for p_id in points3d])
    # colors = np.array([points3d[p_id].rgb for p_id in points3d])
    colors = parser.points_rgb

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        # img_ids = [im.id for im in images.values()]
        # random.shuffle(img_ids)
        img_ids = list(np.arange(nimages))
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position
                    # print(str(Quaternion(client.camera.wxyz).ToR()))
                    # print(str(client.camera.position))

        for img_id in tqdm(img_ids):
            # img = images[img_id]
            # cam = cameras[img.camera_id]

            # Skip images that don't exist.
            # image_filename = images_path / img.name
            # if not image_filename.exists():
            #     continue

            # c2w = img.camtoworld
            c2w = parser.camtoworlds[img_id, :, :].squeeze()

            T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3.from_matrix(c2w[:3, :3]), c2w[:3, 3])

            # T_world_camera = tf.SE3.from_rotation_and_translation(
            #     tf.SO3(img.qvec), img.tvec
            # ).inverse()

            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=10,
                axes_radius=0.05,
            )

            frames.append(frame)

            # # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            # if cam.model != "PINHOLE":
            #     print(f"Expected pinhole camera, but got {cam.model}")

            # H, W = cam.height, cam.width
            cam_id = parser.camera_ids[img_id]

            W, H = parser.imsize_dict[cam_id]
            K = parser.Ks_dict[cam_id]
            # fy = cam.params[1]
            fy = K[1, 1]
            # image = iio.imread(parser.image_paths[img_id])
            # image = image[::downsample_factor, ::downsample_factor]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=4,
                # image=image,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
