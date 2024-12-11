# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import dataclasses
import os
import sys
import threading
import time
import traceback
from typing import Callable, Literal, Optional, Tuple, Union, get_args

import numpy as np
import viser
import viser.transforms as vt

RenderState = Literal["low_move", "low_static", "high"]
RenderAction = Literal["rerender", "move", "static", "update"]


@dataclasses.dataclass
class RenderTask(object):
    action: RenderAction
    camera_state: Optional["CameraState"] = None


class InterruptRenderException(Exception):
    pass


class set_trace_context(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, *_, **__):
        sys.settrace(None)


class Renderer(threading.Thread):
    """This class is responsible for rendering images in the background."""

    def __init__(
        self,
        viewer: "Viewer",
        client: viser.ClientHandle,
        lock: threading.Lock,
    ):
        super().__init__(daemon=True)

        self.viewer = viewer
        self.client = client
        self.lock = lock

        self.running = True
        self.is_prepared_fn = lambda: self.viewer.state.status != "preparing"

        self._render_event = threading.Event()
        self._state: RenderState = "low_static"
        self._task: Optional[RenderTask] = None

        self._target_fps = 30
        self._may_interrupt_render = False

        self._define_transitions()

    def _define_transitions(self):
        transitions: dict[RenderState, dict[RenderAction, RenderState]] = {
            s: {a: s for a in get_args(RenderAction)} for s in get_args(RenderState)
        }
        transitions["low_move"]["static"] = "low_static"
        transitions["low_static"]["static"] = "high"
        transitions["low_static"]["update"] = "high"
        transitions["low_static"]["move"] = "low_move"
        transitions["high"]["move"] = "low_move"
        transitions["high"]["rerender"] = "low_static"
        self.transitions = transitions

    def _may_interrupt_trace(self, frame, event, arg):
        if event == "line":
            if self._may_interrupt_render:
                self._may_interrupt_render = False
                raise InterruptRenderException
        return self._may_interrupt_trace

    def _get_img_wh(self, aspect: float) -> Tuple[int, int]:
        max_img_res = self.viewer._max_img_res_slider.value
        if self._state == "high":
            #  if True:
            H = max_img_res
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        elif self._state in ["low_move", "low_static"]:
            num_view_rays_per_sec = self.viewer.state.num_view_rays_per_sec
            target_fps = self._target_fps
            num_viewer_rays = num_view_rays_per_sec / target_fps
            H = (num_viewer_rays / aspect) ** 0.5
            H = int(round(H, -1))
            H = max(min(max_img_res, H), 30)
            W = int(H * aspect)
            if W > max_img_res:
                W = max_img_res
                H = int(W / aspect)
        else:
            raise ValueError(f"Unknown state: {self._state}.")
        return W, H

    def submit(self, task: RenderTask):
        if self._task is None:
            self._task = task
        elif task.action == "update" and (self._state == "low_move" or self._task.action in ["move", "rerender"]):
            return
        else:
            self._task = task

        if self._state == "high" and self._task.action in ["move", "rerender"]:
            self._may_interrupt_render = True
        self._render_event.set()

    def run(self):
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if not self._render_event.wait(0.2):
                self.submit(RenderTask("static", self.viewer.get_camera_state(self.client)))
            self._render_event.clear()
            task = self._task
            assert task is not None
            #  print(self._state, task.action, self.transitions[self._state][task.action])
            if self._state == "high" and task.action == "static":
                continue
            self._state = self.transitions[self._state][task.action]
            assert task.camera_state is not None
            try:
                with self.lock, set_trace_context(self._may_interrupt_trace):
                    tic = time.time()
                    W, H = img_wh = self._get_img_wh(task.camera_state.aspect)
                    rendered = self.viewer.render_fn(task.camera_state, img_wh)
                    if isinstance(rendered, tuple):
                        img, depth = rendered
                    else:
                        img, depth = rendered, None
                    self.viewer.state.num_view_rays_per_sec = (W * H) / (time.time() - tic)
            except InterruptRenderException:
                continue
            except Exception:
                traceback.print_exc()
                os._exit(1)
            self.client.scene.set_background_image(
                img,
                format="jpeg",
                jpeg_quality=70 if task.action in ["static", "update"] else 40,
                depth=depth,
            )


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh: Tuple[int, int]) -> np.ndarray:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal["rendering", "preparing", "training", "paused", "completed"] = "training"


VIEWER_LOCK = threading.Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                np.ndarray,
                Tuple[np.ndarray, Optional[np.ndarray]],
            ],
        ],
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        if self.mode == "rendering":
            self.state.status = "rendering"

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()

    def _define_guis(self):
        with self.server.gui.add_folder("Stats", visible=self.mode == "training") as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder("Training", visible=self.mode == "training") as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(viewer=self, client=client, lock=self.lock)
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate([vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = train_util * view_time / (train_time - train_util * train_time)
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(RenderTask("update", camera_state))
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
