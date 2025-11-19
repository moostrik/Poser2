# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init


from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.pose.Pose import Pose
from modules.pose.features.Points2D import PointLandmark
from modules.render.renderers import CamImageRenderer

from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase
from modules.utils.PointsAndRects import Rect, Point2f

from modules.utils.HotReloadMethods import HotReloadMethods


class CentreCamLayer(LayerBase):
    def __init__(self, cam_id: int, data_hub: DataHub, type: PoseDataTypes, image_renderer: CamImageRenderer,) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._image_renderer: CamImageRenderer = image_renderer
        self._p_pose: Pose | None = None

        self._safe_eye_midpoint: Point2f = Point2f(0.5, 0.5)
        self._safe_height: float = 0.5
        self._centre_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        self._screen_centre_rect: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        self.data_type: PoseDataTypes = type
        self.target_x: float = 0.5
        self.target_y: float = 0.25
        self.target_height: float = 0.95
        self.dst_aspectratio: float = 9/16


        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def centre_rect(self) -> Rect:
        return self._centre_rect

    @property
    def screen_center_rect(self) -> Rect:
        return self._screen_centre_rect


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self._cam_id

        pose: Pose | None = self._data_hub.get_item(DataType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        if pose.points.get_valid(PointLandmark.left_eye) and pose.points.get_valid(PointLandmark.right_eye):
            left_eye: Point2f = pose.points.get_point2f(PointLandmark.left_eye)
            right_eye: Point2f = pose.points.get_point2f(PointLandmark.right_eye)
            eye_midpoint: Point2f = (left_eye + right_eye) / 2

            self._safe_eye_midpoint = eye_midpoint

        if pose.points.get_valid(PointLandmark.left_hip) and pose.points.get_valid(PointLandmark.right_hip):
            left_hip: Point2f = pose.points.get_point2f(PointLandmark.left_hip)
            right_hip: Point2f = pose.points.get_point2f(PointLandmark.right_hip)
            hip_midpoint: Point2f = (left_hip + right_hip) / 2

            self._safe_height = (hip_midpoint.y - self._safe_eye_midpoint.y)
            self._safe_height *= 2.0  # Rough estimate of full height
            self._safe_height *= self.target_height

        self._centre_rect = self._calculate_centre_rect(pose.bbox.to_rect(), self._safe_eye_midpoint, self._safe_height)
        fbo_rect = Rect(0, 0, self._fbo.width, self._fbo.height)



        # draw_rect
        # print ("Centre rect:", self._centre_rect)

        self._fbo.begin()
        self._image_renderer.draw_roi(fbo_rect, self._centre_rect)

        # draw_rect = Rect(
        #     x=fbo_rect.x - self._centre_rect.x * fbo_rect.width / self._centre_rect.width,
        #     y=fbo_rect.y - self._centre_rect.y * fbo_rect.height / self._centre_rect.height,
        #     width=fbo_rect.width / self._centre_rect.width,
        #     height=fbo_rect.height / self._centre_rect.height
        # )
        # self._image_renderer.draw(draw_rect)

        self._fbo.end()

        self._screen_centre_rect = CentreCamLayer.calculate_screen_center_rect(pose.bbox.to_rect(), self._centre_rect)


    def get_fbo(self) -> Fbo:
        return self._fbo

    def _calculate_centre_rect(self, pose_rect: Rect, centre: Point2f, height: float) -> Rect:
        """Calculate the centered crop rectangle around a pose's center point."""
        centre_world: Point2f = centre * pose_rect.size + pose_rect.position
        width: float = height * self.dst_aspectratio

        return Rect(
            x=centre_world.x - width * self.target_x,
            y=centre_world.y - height * self.target_y,
            width=width,
            height=height
        )

    @staticmethod
    def calculate_screen_center_rect(world_rect: Rect, centre_rect: Rect) -> Rect:
        """
        Calculate the normalized screen space rect for a world rect relative to the centre crop.

        Args:
            world_rect: Any rectangle in world coordinates (e.g., pose.bbox)
            centre_rect: The crop region in world coordinates

        Returns:
            Rect in normalized [0,1] space relative to centre_rect
        """
        return Rect(
            x=(world_rect.x - centre_rect.x) / centre_rect.width,
            y=(world_rect.y - centre_rect.y) / centre_rect.height,
            width=world_rect.width / centre_rect.width,
            height=world_rect.height / centre_rect.height
        )