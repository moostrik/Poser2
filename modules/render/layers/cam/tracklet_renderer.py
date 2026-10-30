# Standard library imports
import numpy as np

# Local application imports
from modules.gl import Text
from ...shaders import DrawColoredRectangle

from modules.camera import CameraTracklet, TrackingStatus
from modules.board import HasCameraTracklets
from ..layer_base import LayerBase



class TrackletRenderer(LayerBase):
    def __init__(self, cam_id: int, board: HasCameraTracklets) -> None:
        self._board: HasCameraTracklets = board
        self._cam_id: int = cam_id
        self._tracklets: list[CameraTracklet] = []
        self._shader: DrawColoredRectangle = DrawColoredRectangle()
        self._text_renderer: Text = Text()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._shader.allocate()
        self._text_renderer.allocate(font_size=14)
        self._width = width
        self._height = height

    def deallocate(self) -> None:
        self._shader.deallocate()
        self._text_renderer.deallocate()

    def draw(self) -> None:
        for tracklet in self._tracklets:
            self.draw_tracklet(tracklet, self._shader, self._text_renderer, self._width, self._height)

    def update(self) -> None:
        tracklets: list[CameraTracklet] | None = self._board.get_camera_tracklets(self._cam_id)
        if tracklets is None:
            self._tracklets = []
        else:
            self._tracklets = tracklets

    def draw_tracklet(self, tracklet: CameraTracklet, shader: DrawColoredRectangle,
                      text_renderer: Text, width: int, height: int) -> None:
        if tracklet.status == TrackingStatus.REMOVED:
            return

        t_x: float = tracklet.roi.x
        t_y: float = tracklet.roi.y
        t_w: float = tracklet.roi.width
        t_h: float = tracklet.roi.height

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == TrackingStatus.REMOVED:
            r, g, b, a = (1.0, 0.0, 0.0, 1.0)

        shader.use(t_x, t_y, t_w, t_h, r, g, b, a)

        # Convert tracklet coordinates to pixel space if needed
        # Tracklet ROI is typically in normalized coordinates (0-1)
        pixel_x: float = t_x * width
        pixel_y: float = t_y * height

        # Draw text at top-left corner of tracklet with padding
        padding: float = 4.0
        text_x: float = pixel_x + padding
        text_y: float = pixel_y + padding

        # Combine ID and Age on one line with labels
        text_string: str = f'ID: {tracklet.id} Age: {tracklet.age}'
        text_width, text_height = text_renderer.measure_text(text_string)

        # Clamp to viewport bounds if text would go outside
        final_x: float = max(0, min(text_x, width - text_width - padding))
        final_y: float = max(0, min(text_y, height - text_height - padding))

        text_renderer.draw_box_text(final_x, final_y, text_string, (1.0, 1.0, 1.0, 1.0),
                                   (0.0, 0.0, 0.0, 0.6), width, height)



