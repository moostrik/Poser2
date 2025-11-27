# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Text import draw_box_string, text_init

from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet

from modules.DataHub import DataHub, DataHubType
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect



class CamDepthTrackRenderer(RendererBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._tracklets: list[DepthTracklet] | None = None
        text_init()

    def allocate(self) -> None:
        pass

    def deallocate(self) -> None:
        pass

    def draw(self, rect: Rect) -> None:
        if self._tracklets is None:
            return
        for depth_tracklet in self._tracklets or []:
            CamDepthTrackRenderer.draw_depth_tracklet(depth_tracklet, rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        self._tracklets: list[DepthTracklet] | None = self._data.get_item(DataHubType.depth_tracklet, self._cam_id)

    @staticmethod
    def draw_depth_tracklet(tracklet: DepthTracklet, x: float, y: float, width: float, height: float) -> None:
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

        glColor4f(r, g, b, a)   # Set color
        glBegin(GL_QUADS)       # Start drawing a quad
        glVertex2f(t_x, t_y)        # Bottom left
        glVertex2f(t_x, t_y + t_h)    # Bottom right
        glVertex2f(t_x + t_w, t_y + t_h)# Top right
        glVertex2f(t_x + t_w, t_y)    # Top left
        glEnd()                 # End drawing
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

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



