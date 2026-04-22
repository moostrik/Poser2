# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Texture import Texture
from modules.gl.Text import Text
from modules.gl import Blit
from modules.render.shaders.cam.DrawColoredRectangle import DrawColoredRectangle

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.board import HasTracklets
from modules.render.layers.LayerBase import LayerBase

from modules.utils.HotReloadMethods import HotReloadMethods

class TrackerPanoramicLayer(LayerBase):
    def __init__(self, board: HasTracklets, num_cams: int, cam_textures: dict[int, Texture] | None = None) -> None:
        self.board: HasTracklets = board
        self.num_cams: int = num_cams
        self._cam_textures: dict[int, Texture] = cam_textures or {}
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

        # Draw per-camera strips as panoramic background
        strip_w = max(1, self.fbo.width // self.num_cams)
        for i in range(self.num_cams):
            if i in self._cam_textures and self._cam_textures[i].allocated:
                glViewport(i * strip_w, 0, strip_w, self.fbo.height)
                Blit.use(self._cam_textures[i])

        # Reset to full FBO viewport for tracklet overlay
        glViewport(0, 0, self.fbo.width, self.fbo.height)

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

            color: list[float] = TrackletIdColor(tracklet.id, aplha=0.9)
            if overlap:
                color[3] = 0.3
            if tracklet.status == TrackingStatus.NEW:
                color = [1.0, 1.0, 1.0, 1.0]

            self._rect_shader.use(roi_x, roi_y, roi_width, roi_height, *color)

            text_x: float = (roi_x * self.fbo.width) + 9
            text_y: float = (roi_y * self.fbo.height) + 22
            self._text.draw_box_text(text_x, text_y, f'W: {world_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
            text_y += 22
            self._text.draw_box_text(text_x, text_y, f'L: {local_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
        self.fbo.end()

