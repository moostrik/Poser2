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

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods
from modules.utils.OneEuroInterpolation import NormalizedEuroInterpolator, OneEuroInterpolator, OneEuroSettings

from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.gl.Mesh import Mesh

class CentreCameraRender(BaseRender):
    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.rect_smoother: PoseSmoothRect = PoseSmoothRect()
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

        if key != 0:
            return

        pose: Pose | None = self.data.get_pose(key, only_new_data=True, consumer_key=self.key())
        if pose is not None:

            if pose.tracklet.is_removed:
                self.rect_smoother.reset()
                self.clear_render()
                self.is_active = False
                return

            if pose.tracklet.is_active:
                self.is_active = True
                self.rect_smoother.add_pose(pose)
                self.last_pose_rect = pose.crop_rect if pose.crop_rect is not None else Rect(0.0, 0.0, 1.0, 1.0)

        if not self.is_active:
            return

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())
        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()


        smooth_pose_rect: Rect = self.rect_smoother.get()
        # print(f"CentreCameraRender: smooth_pose_rect: {smooth_pose_rect}")
        smooth_cam_roi: Rect = CentreCameraRender.pose_rect_to_image_roi(smooth_pose_rect, self.cam_image.width, self.cam_image.height)

        pose_mesh: Mesh = self.pose_meshes.meshes[key]

        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                smooth_cam_roi.x, smooth_cam_roi.y, smooth_cam_roi.width, smooth_cam_roi.height)
        if pose_mesh.isInitialized():
            glLineWidth(10.0)
            glPushMatrix()

            scale_x = self.cam_fbo.width / smooth_cam_roi.width
            scale_y = self.cam_fbo.height / smooth_cam_roi.height
            translate_x = -smooth_cam_roi.x * scale_x
            translate_y = -smooth_cam_roi.y * scale_y

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

    @staticmethod
    def pose_rect_to_image_roi(pose_rect: Rect, image_width: int, image_height: int) -> Rect:
        """Convert a pose rectangle to an image ROI, maintaining aspect ratio."""
        image_aspect_ratio: float = image_width / image_height

        width: float = pose_rect.width / image_aspect_ratio
        x: float =  pose_rect.x + (pose_rect.width - width) / 2.0
        height: float = pose_rect.height
        y: float = pose_rect.y

        return Rect(x, y, width, height)


class PoseSmoothRect():
    def __init__(self, aspectratio: float = 9/16, smoothing_factor: float = 0.8) -> None:
        self.src_aspectratio: float = 16 / 9
        self.dst_aspectratio: float = aspectratio

        self.default_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        self.current_rect: Rect | None = None

        self.settings: OneEuroSettings = OneEuroSettings(25, 1.0, 0.1)

        self.smooth_centre_x: NormalizedEuroInterpolator = NormalizedEuroInterpolator(self.settings)
        self.smooth_centre_y: NormalizedEuroInterpolator = NormalizedEuroInterpolator(self.settings)
        self.smooth_height: OneEuroInterpolator = OneEuroInterpolator(self.settings)

        # Target relative positions
        self.nose_dest_x: float = 0.5
        self.nose_dest_y: float = 0.2
        self.height_dest: float = 0.95

        hot_reload = HotReloadMethods(self.__class__, True, True)


    def add_pose(self, pose: Pose) -> None:
        pose_rect: Rect | None = pose.crop_rect
        pose_points: np.ndarray | None = pose.point_data.points if pose.point_data is not None else None
        pose_height: float | None = pose.measurement_data.length_estimate if pose.measurement_data is not None else None

        if pose_rect is None:
            print(f"PoseSmoothRect: No crop rect for pose {pose.tracklet.id}, this should not happen")
            return
        if pose_rect is None or pose_points is None or pose_height is None:
            return

        nose_x: float = pose_points[PoseJoint.nose.value][0] * pose_rect.width + pose_rect.x
        nose_y: float = pose_points[PoseJoint.nose.value][1] * pose_rect.height + pose_rect.y
        height: float = pose_height * pose_rect.height

        self.smooth_centre_x.add_sample(nose_x)
        self.smooth_centre_y.add_sample(nose_y)
        self.smooth_height.add_sample(height)

    def get(self) -> Rect:
        self.nose_dest_x: float = 0.5
        self.nose_dest_y: float = 0.33
        self.height_dest: float = 0.8

        nose_x: float | None = self.smooth_centre_x.get()
        nose_y: float | None = self.smooth_centre_y.get()
        height: float | None = self.smooth_height.get()

        if nose_x is None or nose_y is None or height is None:
            if self.current_rect is None:
                self.current_rect = self.default_rect
            return self.current_rect

        height = height / self.height_dest
        width: float = height * self.dst_aspectratio

        left: float = nose_x - width * self.nose_dest_x
        top: float = nose_y - height * self.nose_dest_y

        self.current_rect = Rect(left, top, width, height)
        return self.current_rect

    def reset(self) -> None:
        self.current_rect = None

        self.smooth_centre_x.reset()
        self.smooth_centre_y.reset()
        self.smooth_height.reset()
