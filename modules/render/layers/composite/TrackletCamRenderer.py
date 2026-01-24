# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Text import draw_box_string, text_init
from modules.gl import viewport_rect
from modules.render.shaders.cam.DrawColoredQuad import DrawColoredQuad

from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, Rect



class TrackletCamRenderer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._tracklets: list[DepthTracklet] | None = None
        self._shader: DrawColoredQuad = DrawColoredQuad()
        text_init()

    def allocate(self) -> None:
        self._shader.allocate()

    def deallocate(self) -> None:
        self._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        if self._tracklets is None:
            return
        for depth_tracklet in self._tracklets or []:
            TrackletCamRenderer.draw_depth_tracklet(depth_tracklet, rect.x, rect.y, rect.width, rect.height, self._shader)

    def update(self) -> None:
        self._tracklets: list[DepthTracklet] | None = self._data.get_item(DataHubType.depth_tracklet, self._cam_id)

    @staticmethod
    def draw_depth_tracklet(tracklet: DepthTracklet, x: float, y: float, width: float, height: float, shader: DrawColoredQuad) -> None:
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            return

        t_x: float = x + tracklet.roi.x * width
        t_y: float = y + tracklet.roi.y * height
        t_w: float = tracklet.roi.width * width
        t_h: float = tracklet.roi.height* height

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
        if t_x + 66 > width:
            t_x: float = width - 66
        t_y += 22
        string = f'ID: {tracklet.id}'
        draw_box_string(t_x, t_y, string)
        t_y += 22
        string = f'Age: {tracklet.age}'
        draw_box_string(t_x, t_y, string)



