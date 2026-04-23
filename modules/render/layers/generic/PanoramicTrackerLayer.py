# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Text
from ...shaders import DrawColoredRectangle
from ...color_settings import ColorSettings

from modules.tracker import TrackerType, TrackerMetadata
from modules.tracker import Tracklet, TrackingStatus

from modules.board import HasTracklets
from ..LayerBase import LayerBase

from modules.utils import HotReloadMethods

class PanoramicTrackerLayer(LayerBase):
    def __init__(self, board: HasTracklets, num_cams: int, color_settings: ColorSettings) -> None:
        self.board: HasTracklets = board
        self.num_cams: int = num_cams
        self._color_settings: ColorSettings = color_settings
        self.fbo: Fbo = Fbo()
        self._text: Text = Text()
        self._rect_shader: DrawColoredRectangle = DrawColoredRectangle()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self.fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self._text.allocate()
        self._rect_shader.allocate()

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self._text.deallocate()
        self._rect_shader.deallocate()

    def update(self) -> None:
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        tracklets: dict[int, Tracklet] = self.board.get_tracklets()
        if not tracklets:
            self.fbo.end()
            return

        for tracklet in tracklets.values():
            if tracklet is None:
                continue
            if tracklet.status != TrackingStatus.TRACKED and tracklet.status != TrackingStatus.NEW:
                continue

            tracklet_metadata: TrackerMetadata | None = tracklet.metadata
            if tracklet_metadata is None or tracklet_metadata.tracker_type != TrackerType.PANORAMIC:
                continue

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            local_angle: float = getattr(tracklet.metadata, "local_angle", 0.0)
            overlap: bool = getattr(tracklet.metadata, "overlap", False)

            roi_width: float = tracklet.roi.width / self.num_cams
            roi_height: float = tracklet.roi.height
            roi_x: float = world_angle / 360.0 - roi_width / 2.0
            roi_y: float = tracklet.roi.y

            colors = self._color_settings.track_color_tuples
            r, g, b, a = colors[tracklet.id % len(colors)]
            if overlap:
                a = 0.3
            color = (r, g, b, a)

            self._rect_shader.use(roi_x, roi_y, roi_width, roi_height, *color)

            text_x: float = (roi_x * self.fbo.width) + 9
            text_y: float = (roi_y * self.fbo.height) + 22
            self._text.draw_box_text(text_x, text_y, f'W: {world_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
            text_y += 22
            self._text.draw_box_text(text_x, text_y, f'L: {local_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
        self.fbo.end()

