# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Texture import Texture
from modules.gl.Text import Text

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.board import HasTracklets
from modules.render.layers.LayerBase import LayerBase

from modules.utils.HotReloadMethods import HotReloadMethods

class TrackerPanoramicLayer(LayerBase):
    def __init__(self, board: HasTracklets, num_cams: int) -> None:
        self.board: HasTracklets = board
        self.num_cams: int = num_cams
        self.fbo: Fbo = Fbo()
        self._text: Text = Text()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self.fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self._text.allocate()

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        tracklets: dict[int, Tracklet] = self.board.get_tracklets()
        if tracklets is None:
            return
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

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

            roi_width: float = tracklet.roi.width * self.fbo.width / self.num_cams
            roi_height: float = tracklet.roi.height * self.fbo.height
            roi_x: float = world_angle / 360.0 * self.fbo.width - roi_width / 2.0
            roi_y: float = tracklet.roi.y * self.fbo.height

            color: list[float] = TrackletIdColor(tracklet.id, aplha=0.9)
            if overlap == True:
                color[3] = 0.3
            if tracklet.status == TrackingStatus.NEW:
                color = [1.0, 1.0, 1.0, 1.0]

            glColor4f(*color)
            glBegin(GL_QUADS)       # Start drawing a quad
            glVertex2f(roi_x, roi_y)        # Bottom left
            glVertex2f(roi_x, roi_y + roi_height)    # Bottom right
            glVertex2f(roi_x + roi_width, roi_y + roi_height)# Top right
            glVertex2f(roi_x + roi_width, roi_y)    # Top left
            glEnd()                 # End drawing

            roi_x += 9
            roi_y += 22
            self._text.draw_box_text(roi_x, roi_y, f'W: {world_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
            roi_y += 22
            self._text.draw_box_text(roi_x, roi_y, f'L: {local_angle:.1f}', (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), self.fbo.width, self.fbo.height)
        self.fbo.end()

