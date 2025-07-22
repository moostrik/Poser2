# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Text import draw_box_string, text_init

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class OnePerCamRender(BaseRender):
    def __init__(self, data: DataManager) -> None:
        self.data: DataManager = data
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
        pass