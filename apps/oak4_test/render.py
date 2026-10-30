"""The OAK4 test app's render: the camera image with the tracklets, the boxes and the skeletons, and the
camera's readings over it, filling the window."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA

from modules.gl import RenderBase, Shader, Style, clear_color
from modules.camera import RigCameraSettings, CameraCheckSettings
from modules.render import layers as ls

from .render_board import RenderBoard
from .settings import Layers, RenderSettings


INTERFACE_LAYERS: list[Layers] = [Layers.tracker, Layers.cam_readings]


class Oak4TestRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings, camera: RigCameraSettings,
                 camera_check: CameraCheckSettings) -> None:
        super().__init__(settings.window)
        self.settings: RenderSettings = settings
        self.board: RenderBoard = board

        cam_image = ls.ImageSourceLayer(0, self.board)
        self.L: dict[Layers, ls.LayerBase] = {
            Layers.cam_image:    cam_image,
            Layers.tracker:      ls.TrackerCompositor(0, self.board, cam_image.texture, settings.tracker, settings.colors),
            Layers.cam_readings: ls.CameraReadingsLayer(camera, camera_check),
        }

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.allocate_window_renders(width, height)

    def allocate(self) -> None:
        self.L[Layers.cam_image].allocate(1080, 1080, GL_RGBA16F)
        self.allocate_window_renders(self.settings.window.width, self.settings.window.height)
        Shader.enable_hot_reload()

    def allocate_window_renders(self, width: int, height: int) -> None:
        for layer_type in INTERFACE_LAYERS:
            self.L[layer_type].allocate(width, height, GL_RGBA)

    def deallocate(self) -> None:
        for layer in self.L.values():
            layer.deallocate()

    def update(self) -> None:
        self._notify_update()
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for layer in self.L.values():
            layer.update()

    def draw_main(self, width: int, height: int) -> None:
        clear_color()
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for layer_type in self.settings.layer.final:
            self.L[layer_type].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        clear_color()
