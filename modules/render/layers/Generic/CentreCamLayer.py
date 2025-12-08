# Standard library imports
import numpy as np
import math

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D, PointLandmark
from modules.render.layers.renderers import CamImageRenderer, CamMaskRenderer
from modules.utils.PointsAndRects import Rect, Point2f

# GL
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.LayerBase import LayerBase
from modules.gl.shaders.Blend import Blend
from modules.gl.shaders.DrawRoi import DrawRoi
from modules.gl.shaders.PosePointLines import PosePointLines

from modules.utils.HotReloadMethods import HotReloadMethods


class CentreCamLayer(LayerBase):
    _blend_shader = Blend()
    _roi_shader = DrawRoi()
    _point_shader = PosePointLines()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 cam_image: CamImageRenderer, cam_mask: CamMaskRenderer) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._crop_fbo: Fbo = Fbo()
        self._mask_fbo: Fbo = Fbo()
        self._blend_fbo: SwapFbo = SwapFbo()
        self._point_fbo: Fbo = Fbo()
        self._cam_image: CamImageRenderer = cam_image
        self._cam_mask: CamMaskRenderer = cam_mask
        self._p_pose: Frame | None = None

        self._eye_midpoint: Point2f = Point2f(0.5, 0.25)
        self._hip_midpoint: Point2f = Point2f(0.5, 0.6)
        self._rotation: float = 0.0
        self._crop_roi: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        self.data_type: PoseDataHubTypes = data_type
        self.target_eye: Point2f = Point2f(0.5, 0.25)
        self.target_hip: Point2f = Point2f(0.5, 0.6)

        self.dst_aspectratio: float = 9/16
        self.blend_factor: float = 0.25

        self.transformed_points: Points2D = Points2D.create_dummy()

        self._on_points_updated = None

        HotReloadMethods(self.__class__, True, True)

    def set_points_callback(self, callback) -> None:
        self._on_points_updated = callback

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._crop_fbo.allocate(width, height, internal_format)
        self._mask_fbo.allocate(width, height, internal_format)
        self._blend_fbo.allocate(width, height, internal_format)
        self._point_fbo.allocate(width, height, internal_format)
        if not CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.allocate(monitor_file=True)
        if not CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._crop_fbo.deallocate()
        self._mask_fbo.deallocate()
        self._blend_fbo.deallocate()
        self._point_fbo.deallocate()
        if CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.deallocate()
        if CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.deallocate()
        if CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._blend_fbo.draw(rect.x, rect.y, rect.width, rect.height)

        # return
        self.draw_points(rect)

        # Debug: draw target positions
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glVertex2f(rect.x + self.target_eye.x * rect.width,
                   rect.y + self.target_eye.y * rect.height)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glVertex2f(rect.x + self.target_hip.x * rect.width,
                   rect.y + self.target_hip.y * rect.height)
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def draw_points(self, rect: Rect) -> None:
        self._point_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.allocate(monitor_file=True)
        if not CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.allocate(monitor_file=True)

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

        if pose is self._p_pose:
            return
        self._p_pose = pose

        LayerBase.setView(self._blend_fbo.width, self._blend_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._crop_fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._point_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return

        # Update eye midpoint
        if pose.points.get_valid(PointLandmark.left_eye) and pose.points.get_valid(PointLandmark.right_eye):
            left_eye: Point2f = pose.points.get_point2f(PointLandmark.left_eye)
            right_eye: Point2f = pose.points.get_point2f(PointLandmark.right_eye)
            self._eye_midpoint = (left_eye + right_eye) / 2

        # Update hip midpoint
        if pose.points.get_valid(PointLandmark.left_hip) and pose.points.get_valid(PointLandmark.right_hip):
            left_hip: Point2f = pose.points.get_point2f(PointLandmark.left_hip)
            right_hip: Point2f = pose.points.get_point2f(PointLandmark.right_hip)
            self._hip_midpoint = (left_hip + right_hip) / 2

        aspect: float = self._cam_image.width / self._cam_image.height
        bbox: Rect = pose.bbox.to_rect()

        # Convert from bbox-relative to texture coordinates
        eye: Point2f = self._eye_midpoint * bbox.size + bbox.position
        hip: Point2f = self._hip_midpoint * bbox.size + bbox.position

        # Calculate rotation (with aspect correction to match shader)
        delta: Point2f = hip - eye
        self._rotation = math.atan2(delta.x * aspect, delta.y)

        # Calculate crop ROI
        distance: float = math.hypot(delta.x * aspect, delta.y)
        target_distance: float = self.target_hip.y - self.target_eye.y
        height: float = distance / target_distance
        width: float = height * self.dst_aspectratio

        self._crop_roi = Rect(
            x=eye.x - width * self.target_eye.x,
            y=eye.y - height * self.target_eye.y,
            width=width,
            height=height
        )

        # Render cropped/rotated image
        CentreCamLayer._roi_shader.use(self._crop_fbo.fbo_id, self._cam_image.tex_id, self._crop_roi, self._rotation, eye, False, True)

        # Calculate mask ROI (mask is in bbox-space [0,1], which is square)
        # We need to account for bbox aspect ratio to match the texture space
        bbox_aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0

        mask_delta = self._hip_midpoint - self._eye_midpoint
        # Apply bbox aspect correction to match what shader will do with mask texture
        mask_rotation = math.atan2(mask_delta.x * bbox_aspect, mask_delta.y)
        mask_distance = math.hypot(mask_delta.x * bbox_aspect, mask_delta.y)
        mask_height = mask_distance / target_distance
        mask_width = mask_height * bbox_aspect

        mask_crop_roi = Rect(
            x=self._eye_midpoint.x - mask_width * self.target_eye.x,
            y=self._eye_midpoint.y - mask_height * self.target_eye.y,
            width=mask_width,
            height=mask_height
        )

        # Render mask to separate FBO
        CentreCamLayer._roi_shader.use(self._mask_fbo.fbo_id, self._cam_mask.tex_id, mask_crop_roi, mask_rotation, self._eye_midpoint, False, True)



        # Blend with previous frame
        self._blend_fbo.swap()
        CentreCamLayer._blend_shader.use(self._blend_fbo.fbo_id, self._blend_fbo.back_tex_id,self._mask_fbo.tex_id, self.blend_factor)

        # Transform points
        self._transformed_points: Points2D = CentreCamLayer._transform_points(pose.points, bbox, eye, aspect, self._rotation, self._crop_roi)

        if self._on_points_updated is not None:
            self._on_points_updated(self._transformed_points)

        # return
        # Render points
        line_width: float = 50.0 / self._point_fbo.height
        line_smooth: float = 25.0 / self._point_fbo.height
        self._point_fbo.clear(0.0, 0.0, 0.0, 0.0)
        CentreCamLayer._point_shader.use(self._point_fbo.fbo_id, self._transformed_points, line_width=line_width, line_smooth=line_smooth)

    @staticmethod
    def _transform_points(points: Points2D, bbox: Rect, eye: Point2f,
                          aspect: float, rotation: float, crop_roi: Rect) -> Points2D:
        """Transform pose points to crop-space coordinates [0,1]."""
        x_bbox, y_bbox = points.get_xy_arrays()

        # Convert from bbox-relative to texture coordinates
        x = x_bbox * bbox.width + bbox.x
        y = y_bbox * bbox.height + bbox.y

        # Rotate around eye with aspect correction (matches shader)
        dx = (x - eye.x) * aspect
        dy = y - eye.y

        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        x_rot = (cos_a * dx - sin_a * dy) / aspect + eye.x
        y_rot = sin_a * dx + cos_a * dy + eye.y

        # Convert to crop ROI space
        x_crop = (x_rot - crop_roi.x) / crop_roi.width
        y_crop = (y_rot - crop_roi.y) / crop_roi.height

        return Points2D.from_xy_arrays(x_crop, y_crop, points.scores)