# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.Pose import Pose
from modules.pose.features.Point2DFeature import PointLandmark
from modules.data.depricated.RenderDataHub_old import RenderDataHub_Old

from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.render.meshes.PoseMesh import PoseMesh

from modules.gl.Mesh import Mesh

class CentreCamLayer(LayerBase):
    def __init__(self, data: DataHub, cam_id: int) -> None:
        self.data: DataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.is_active: bool = False

        self.last_pose_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.cam_fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.cam_fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self.cam_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id

        pose: Pose | None = self.data.get_smooth_pose(key, only_new_data=True, consumer_key=self.data_consumer_key)
        if pose is None:
            return

        self.last_pose_rect = pose.bbox.to_rect()

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.data_consumer_key)
        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        smooth_pose_rect: Rect | None = pose.bbox.to_rect()
        if smooth_pose_rect is None:
            return

        LayerBase.setView(self.cam_fbo.width, self.cam_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.cam_fbo.begin()
        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                smooth_pose_rect.x, smooth_pose_rect.y, smooth_pose_rect.width, smooth_pose_rect.height)
        self.cam_fbo.end()


    def clear_render(self) -> None:
        LayerBase.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.cam_fbo.end()

    def get_fbo(self) -> Fbo:
        return self.cam_fbo
