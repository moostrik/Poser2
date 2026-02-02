# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Text import draw_box_string, text_init
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
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._shader.allocate()

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self) -> None:
        if self._tracklets is None:
            return
        for depth_tracklet in self._tracklets or []:
            TrackletRenderer.draw_depth_tracklet(depth_tracklet, self._shader)

    def update(self) -> None:
        self._tracklets: list[DepthTracklet] | None = self._data.get_item(DataHubType.depth_tracklet, self._cam_id)

    @staticmethod
    def draw_depth_tracklet(tracklet: DepthTracklet, shader: DrawColoredRectangle) -> None:
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
        t_x += t_w -6
        t_y += 22
        string = f'ID: {tracklet.id}'
        draw_box_string(t_x, t_y, string)
        t_y += 22
        string = f'Age: {tracklet.age}'
        draw_box_string(t_x, t_y, string)



