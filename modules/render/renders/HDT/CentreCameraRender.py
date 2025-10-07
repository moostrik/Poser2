# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothData import PoseSmoothData

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.render.meshes.PoseMeshes import PoseMeshes

from modules.gl.Mesh import Mesh

class CentreCameraRender(BaseRender):
    def __init__(self, data: DataManager, smooth_data: PoseSmoothData, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.smooth_data: PoseSmoothData = smooth_data
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.is_active: bool = False

        self.pose_meshes: PoseMeshes = pose_meshes
        self.last_pose_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        # self.last_pose: Pose | None = None
        # self.last_Rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
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

        # if key != 0:
        #     return

        pose: Pose | None = self.data.get_pose(key, only_new_data=True, consumer_key=self.key())
        if pose is not None:
            # print(f"CentreCameraRender: Updating render for pose {pose.tracklet.is_removed}")

            if pose.tracklet.is_removed:
                # print(f"CentreCameraRender: Pose {pose.tracklet.id} removed, clearing render")
                self.clear_render()
                self.is_active = False
                return

            if pose.tracklet.is_active:
                self.is_active = True
                self.last_pose_rect = pose.crop_rect if pose.crop_rect is not None else Rect(0.0, 0.0, 1.0, 1.0)

        if not self.is_active:
            return

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())
        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        smooth_pose_rect: Rect | None = self.smooth_data.get_smoothed_rect(key)
        if smooth_pose_rect is None:
            return

        pose_mesh: Mesh = self.pose_meshes.meshes[key]

        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        glClear(GL_COLOR_BUFFER_BIT)
        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                smooth_pose_rect.x, smooth_pose_rect.y, smooth_pose_rect.width, smooth_pose_rect.height)

        if pose_mesh.isInitialized():

            glLineWidth(10.0)
            glPushMatrix()

            scale_x = self.cam_fbo.width / smooth_pose_rect.width
            scale_y = self.cam_fbo.height / smooth_pose_rect.height
            translate_x = -smooth_pose_rect.x * scale_x
            translate_y = -smooth_pose_rect.y * scale_y

            # Apply transformation
            glTranslatef(translate_x, translate_y, 0.0)
            glScalef(scale_x, scale_y, 1.0)

            # Now draw the mesh with the last_pose_rect coordinates
            # This maps from pose_rect to the transformed space
            pose_mesh.draw(
                self.last_pose_rect.x,
                self.last_pose_rect.y,
                self.last_pose_rect.width,
                self.last_pose_rect.height
            )

            glPopMatrix()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.cam_fbo.end()


    def clear_render(self) -> None:
        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.cam_fbo.end()

    def get_fbo(self) -> Fbo:
        return self.cam_fbo
