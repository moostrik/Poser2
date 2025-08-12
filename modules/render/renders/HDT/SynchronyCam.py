# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.PoseStream import PoseStreamData
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.pose.PoseDefinitions import JointAngle, Keypoint, Pose, PoseAngleNames
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.gl.shaders.HD_Sync import HD_Sync

from modules.utils.HotReloadMethods import HotReloadMethods

class SynchronyCam(BaseRender):

    shader = HD_Sync()
    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.movement: float = 0.0
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        if not SynchronyCam.shader.allocated:
            SynchronyCam.shader.allocate(True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        if SynchronyCam.shader.allocated:
            SynchronyCam.shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self, cam_fbo: list[Fbo]) -> None:
        key: int = self.cam_id

        # syncs: PairCorrelationStreamData | None = self.data.get_correlation_streams(False, self.key())
        # if syncs is None:
        #     return

        BaseRender.setView(self.fbo.width, self.fbo.height)

        if not SynchronyCam.shader.allocated:
            SynchronyCam.shader.allocate(True)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()




        SynchronyCam.shader.use(self.fbo.fbo_id, cam_fbo[0].tex_id, cam_fbo[1].tex_id, cam_fbo[2].tex_id, 1.0, 1.0)




