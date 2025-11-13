# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Text import draw_box_string, text_init

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class TrackerPanoramicLayer(LayerBase):
    def __init__(self, data: DataHub, num_cams: int) -> None:
        self.data: DataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.num_cams: int = num_cams
        self.fbo: Fbo = Fbo()
        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        tracklets: dict[int, Tracklet] = self.data.get_tracklets()
        if tracklets is None:
            return
        # print(f"PanoramicTrackerRender: Updating with {len(tracklets)} tracklets")
        LayerBase.setView(self.fbo.width, self.fbo.height)
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

            glColor4f(*color)  # Reset color
            glBegin(GL_QUADS)       # Start drawing a quad
            glVertex2f(roi_x, roi_y)        # Bottom left
            glVertex2f(roi_x, roi_y + roi_height)    # Bottom right
            glVertex2f(roi_x + roi_width, roi_y + roi_height)# Top right
            glVertex2f(roi_x + roi_width, roi_y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

            string: str
            roi_x += 9
            roi_y += 22
            string = f'W: {world_angle:.1f}'
            draw_box_string(roi_x, roi_y, string)
            roi_y += 22
            string = f'L: {local_angle:.1f}'
            draw_box_string(roi_x, roi_y, string)
        self.fbo.end()

