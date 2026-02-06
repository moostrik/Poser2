# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Text
from modules.render.shaders import DrawColoredRectangle

from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, Rect



class TrackletRenderer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._tracklets: list[DepthTracklet] | None = None
        self._shader: DrawColoredRectangle = DrawColoredRectangle()
        self._text_renderer: Text = Text()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._shader.allocate()
        self._text_renderer.allocate("files/RobotoMono-Regular.ttf", font_size=14)
        self._width = width
        self._height = height

    def deallocate(self) -> None:
        self._shader.deallocate()
        self._text_renderer.deallocate()

    def draw(self) -> None:
        if self._tracklets is None:
            return
        for depth_tracklet in self._tracklets or []:
            self.draw_depth_tracklet(depth_tracklet, self._shader, self._text_renderer, self._width, self._height)

    def update(self) -> None:
        self._tracklets: list[DepthTracklet] | None = self._data.get_item(DataHubType.depth_tracklet, self._cam_id)

    def draw_depth_tracklet(self, tracklet: DepthTracklet, shader: DrawColoredRectangle,
                           text_renderer: Text, width: int, height: int) -> None:
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            return

        t_x: float = tracklet.roi.x
        t_y: float = tracklet.roi.y
        t_w: float = tracklet.roi.width
        t_h: float = tracklet.roi.height

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == DepthTracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == DepthTracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            r, g, b, a = (1.0, 0.0, 0.0, 1.0)

        shader.use(t_x, t_y, t_w, t_h, r, g, b, a)

        string: str
        t_x += t_w - 6
        t_y += 22
        string = f'ID: {tracklet.id}'
        text_renderer.draw_box_text(t_x, t_y, string, (1.0, 1.0, 1.0, 1.0),
                                   (0.0, 0.0, 0.0, 0.6), width, height)
        t_y += 22
        string = f'Age: {tracklet.age}'
        text_renderer.draw_box_text(t_x, t_y, string, (1.0, 1.0, 1.0, 1.0),
                                   (0.0, 0.0, 0.0, 0.6), width, height)



