

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.render.Draw.DrawBase import DrawBase
from modules.gl.Text import draw_string, draw_box_string, text_init


class DrawPanoramicTracker(DrawBase):
    """Methods for updating meshes based on pose data."""
    def __init__(self, data: DataManager, num_cams: int) -> None:
        self.data: DataManager = data
        self.num_cams: int = num_cams
        self.fbo: Fbo = Fbo()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.fbo.deallocate()

    def update(self, only_if_dirty: bool) -> None:
        tracklets: dict[int, Tracklet] = self.data.get_tracklets()
        if tracklets is not None:
            self.draw_map_positions(tracklets, self.fbo, self.num_cams)

    def draw(self, x: float, y: float, width: float, height: float) -> None:
        self.fbo.draw(x, y, width, height)

    @staticmethod
    def draw_map_positions(tracklets: dict[int, Tracklet], fbo: Fbo, num_cams: int) -> None:
        fbo.begin()
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

            roi_width: float = tracklet.roi.width * fbo.width / num_cams
            roi_height: float = tracklet.roi.height * fbo.height
            roi_x: float = world_angle / 360.0 * fbo.width
            roi_y: float = tracklet.roi.y * fbo.height

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

        glFlush()  # Render now
        fbo.end()

