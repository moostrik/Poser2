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
from modules.gl.Texture import Texture
from modules.gl.LayerBase import LayerBase
from modules.gl.shaders.DrawRoi import DrawRoi
from modules.gl.shaders.Blend import Blend
from modules.gl.shaders.MaskBlend import MaskBlend
from modules.gl.shaders.MaskAA import MaskAA
from modules.gl.shaders.MaskBlur import MaskBlur
from modules.gl.shaders.ApplyMask import ApplyMask
from modules.gl.shaders.PosePointLines import PosePointLines

from modules.utils.HotReloadMethods import HotReloadMethods


class CentreCamLayer(LayerBase):
    _blend_shader = Blend()

    _mask_blur_shader = MaskBlur()
    _mask_blend_shader = MaskBlend()
    _mask_AA_shader = MaskAA()

    _mask_shader = ApplyMask()

    _roi_shader = DrawRoi()
    _point_shader = PosePointLines()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 cam_image: CamImageRenderer, cam_mask: CamMaskRenderer) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._fbo = Fbo()
        self._cam_fbo: Fbo = Fbo()
        self._mask_fbo: Fbo = Fbo()
        self._cam_blend_fbo: SwapFbo = SwapFbo()
        self._mask_blend_fbo: SwapFbo = SwapFbo()
        self._mask_AA_fbo: Fbo = Fbo()
        self._mask_blur_fbo: SwapFbo = SwapFbo()
        self._point_fbo: Fbo = Fbo()
        self._cam_image: CamImageRenderer = cam_image
        self._cam_mask: CamMaskRenderer = cam_mask
        self._p_pose: Frame | None = None

        self._shoulder_midpoint: Point2f = Point2f(0.5, 0.3)
        self._hip_midpoint: Point2f = Point2f(0.5, 0.6)
        self._rotation: float = 0.0
        self._crop_roi: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        self.data_type: PoseDataHubTypes = data_type
        self.target_top: Point2f = Point2f(0.5, 0.3)
        self.target_bottom: Point2f = Point2f(0.5, 0.6)

        self.dst_aspectratio: float = 9/16
        self.blend_factor: float = 0.25

        self.transformed_points: Points2D = Points2D.create_dummy()

        self._on_points_updated = None

        HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def cam_texture(self) -> Texture:
        return self._cam_blend_fbo.texture

    @property
    def mask_texture(self) -> Texture:
        return self._mask_blur_fbo.texture

    def set_points_callback(self, callback) -> None:
        self._on_points_updated = callback

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._cam_fbo.allocate(width, height, internal_format)
        self._cam_blend_fbo.allocate(width, height, internal_format)
        self._mask_fbo.allocate(width, height, GL_R32F)
        self._mask_blend_fbo.allocate(width, height, GL_R32F)
        self._mask_AA_fbo.allocate(width, height, GL_R32F)
        self._mask_blur_fbo.allocate(width, height, GL_R32F)

        self._point_fbo.allocate(width, height, internal_format)
        if not CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.allocate(monitor_file=True)
        if not CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._cam_fbo.deallocate()
        self._cam_blend_fbo.deallocate()

        self._mask_fbo.deallocate()
        self._mask_blend_fbo.deallocate()
        self._mask_AA_fbo.deallocate()
        self._mask_blur_fbo.deallocate()

        self._point_fbo.deallocate()
        if CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.deallocate()
        if CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.deallocate()
        if CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        # Enable blending for drawing to screen with straight alpha

        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._cam_blend_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._mask_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._mask_blend_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._mask_AA_fbo.draw(rect.x, rect.y, rect.width, rect.height)

        # return
        self.draw_points(rect)

        # Debug: draw target positions
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glVertex2f(rect.x + self.target_top.x * rect.width,
                   rect.y + self.target_top.y * rect.height)
        glColor4f(0.0, 0.0, 0.0, 1.0)
        glVertex2f(rect.x + self.target_bottom.x * rect.width,
                   rect.y + self.target_bottom.y * rect.height)
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def draw_points(self, rect: Rect) -> None:
        self._point_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # Ensure shaders are allocated
        if not CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.allocate(monitor_file=True)
        if not CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.allocate(monitor_file=True)
        if not CentreCamLayer._mask_blend_shader.allocated:
            CentreCamLayer._mask_blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._mask_shader.allocated:
            CentreCamLayer._mask_shader.allocate(monitor_file=True)
        if not CentreCamLayer._mask_AA_shader.allocated:
            CentreCamLayer._mask_AA_shader.allocate(monitor_file=True)
        if not CentreCamLayer._mask_blur_shader.allocated:
            CentreCamLayer._mask_blur_shader.allocate(monitor_file=True)

        # Get pose data
        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)
        if pose is self._p_pose:
            return
        self._p_pose = pose

        # Setup GL state
        LayerBase.setView(self._cam_blend_fbo.width, self._cam_blend_fbo.height)
        # Disable blending during FBO rendering - shaders handle all compositing
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._cam_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._mask_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        if pose is None:
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._cam_blend_fbo.swap()
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

            self._mask_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._mask_blend_fbo.swap()
            self._mask_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

            self._point_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return

        # Update body landmarks
        shoulder_mid = CentreCamLayer._get_midpoint(pose.points, PointLandmark.left_shoulder, PointLandmark.right_shoulder)
        if shoulder_mid:
            self._shoulder_midpoint = shoulder_mid
        hip_mid = CentreCamLayer._get_midpoint(pose.points, PointLandmark.left_hip, PointLandmark.right_hip)
        if hip_mid:
            self._hip_midpoint = hip_mid

        bbox: Rect = pose.bbox.to_rect()
        target_distance: float = self.target_bottom.y - self.target_top.y

        # Render camera image with ROI
        cam_top: Point2f = self._shoulder_midpoint * bbox.size + bbox.position
        cam_bot: Point2f = self._hip_midpoint * bbox.size + bbox.position
        cam_aspect: float = self._cam_image.width / self._cam_image.height

        self._rotation, _, self._crop_roi = CentreCamLayer._calculate_roi(
            cam_top, cam_bot, self.target_top, target_distance, self.dst_aspectratio
        )
        CentreCamLayer._roi_shader.use(self._cam_fbo.fbo_id, self._cam_image.tex_id,
                                      self._crop_roi, self._rotation, cam_top, cam_aspect, False, True)


        self.blend_factor = 0.5
        self._cam_blend_fbo.swap()
        CentreCamLayer._blend_shader.use(self._cam_blend_fbo.fbo_id, self._cam_blend_fbo.back_tex_id,self._cam_fbo.tex_id, self.blend_factor)


        # Render mask with ROI
        bbox_aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0

        # Calculate mask rotation with aspect correction (matching old working version)
        mask_delta = self._hip_midpoint - self._shoulder_midpoint
        mask_rotation = math.atan2(mask_delta.x * bbox_aspect, mask_delta.y)
        mask_distance = math.hypot(mask_delta.x * bbox_aspect, mask_delta.y)
        mask_height = mask_distance / target_distance
        mask_width = mask_height * bbox_aspect

        mask_crop_roi = Rect(
            x=self._shoulder_midpoint.x - mask_width * self.target_top.x,
            y=self._shoulder_midpoint.y - mask_height * self.target_top.y,
            width=mask_width,
            height=mask_height
        )

        CentreCamLayer._roi_shader.use(self._mask_fbo.fbo_id, self._cam_mask.tex_id,
                                      mask_crop_roi, mask_rotation, self._shoulder_midpoint, bbox_aspect, False, True)

        # Blend frames with mask upscaling and blur
        self.blend_factor = 0.33
        self._mask_blend_fbo.swap()
        CentreCamLayer._mask_blend_shader.use(self._mask_blend_fbo.fbo_id, self._mask_blend_fbo.back_tex_id, self._mask_fbo.tex_id, self.blend_factor)
        CentreCamLayer._mask_AA_shader.use(self._mask_AA_fbo.fbo_id, self._mask_blend_fbo.tex_id)

        self._mask_blur_fbo.begin()
        self._mask_AA_fbo.draw(0, 0, self._mask_blur_fbo.width, self._mask_blur_fbo.height)
        self._mask_blur_fbo.end()

        blur_steps = 0
        blur_radius = 8.0
        for i in range(blur_steps):
            self._mask_blur_fbo.swap()
            CentreCamLayer._mask_blur_shader.use(self._mask_blur_fbo.fbo_id, self._mask_blur_fbo.back_tex_id, True, blur_radius)
            self._mask_blur_fbo.swap()
            CentreCamLayer._mask_blur_shader.use(self._mask_blur_fbo.fbo_id, self._mask_blur_fbo.back_tex_id, False, blur_radius)

        CentreCamLayer._mask_shader.use(self._fbo.fbo_id, self._cam_blend_fbo.tex_id, self._mask_blur_fbo.tex_id)



        # Transform and render pose points
        self._transformed_points = CentreCamLayer._transform_points(
            pose.points, bbox, cam_top, cam_aspect, self._rotation, self._crop_roi
        )
        if self._on_points_updated is not None:
            self._on_points_updated(self._transformed_points)

        line_width: float = 50.0 / self._point_fbo.height
        line_smooth: float = 25.0 / self._point_fbo.height
        self._point_fbo.clear(0.0, 0.0, 0.0, 0.0)
        CentreCamLayer._point_shader.use(self._point_fbo.fbo_id, self._transformed_points,
                                        line_width=line_width, line_smooth=line_smooth)


        glEnable(GL_BLEND)

    @staticmethod
    def _get_midpoint(points: Points2D, left_landmark: PointLandmark, right_landmark: PointLandmark) -> Point2f | None:
        """Get midpoint between two landmarks if both are valid.

        Returns: Midpoint or None if either landmark is invalid
        """
        if points.get_valid(left_landmark) and points.get_valid(right_landmark):
            left = points.get_point2f(left_landmark)
            right = points.get_point2f(right_landmark)
            return (left + right) / 2
        return None

    @staticmethod
    def _calculate_roi(top_point: Point2f, bottom_point: Point2f,
                      target_top: Point2f, target_distance: float,
                      output_aspect: float) -> tuple[float, float, Rect]:
        """Calculate rotation angle and ROI for cropping.

        Returns: (rotation_angle, distance, roi_rect)
        """
        delta = bottom_point - top_point
        rotation = math.atan2(delta.x, delta.y)
        distance = math.hypot(delta.x, delta.y)

        height = distance / target_distance
        width = height * output_aspect

        roi = Rect(
            x=top_point.x - width * target_top.x,
            y=top_point.y - height * target_top.y,
            width=width,
            height=height
        )

        return rotation, distance, roi

    @staticmethod
    def _transform_points(points: Points2D, bbox: Rect, top: Point2f,
                          aspect: float, rotation: float, crop_roi: Rect) -> Points2D:
        """Transform pose points to crop-space coordinates [0,1]."""
        x_bbox, y_bbox = points.get_xy_arrays()

        # Convert from bbox-relative to texture coordinates
        x = x_bbox * bbox.width + bbox.x
        y = y_bbox * bbox.height + bbox.y

        # Rotate around point with aspect correction (matches shader behavior)
        dx = (x - top.x) * aspect
        dy = y - top.y

        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        x_rot = (cos_a * dx - sin_a * dy) / aspect + top.x
        y_rot = sin_a * dx + cos_a * dy + top.y

        # Convert to crop ROI space
        x_crop = (x_rot - crop_roi.x) / crop_roi.width
        y_crop = (y_rot - crop_roi.y) / crop_roi.height

        return Points2D.from_xy_arrays(x_crop, y_crop, points.scores)