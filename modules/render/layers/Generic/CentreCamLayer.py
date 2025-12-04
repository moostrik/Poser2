# Standard library imports
import numpy as np
import math

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init


from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D, PointLandmark
from modules.render.layers.renderers import CamImageRenderer

from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase
from modules.utils.PointsAndRects import Rect, Point2f

# Shaders
from modules.gl.shaders.Blend import Blend
from modules.gl.shaders.DrawRoi import DrawRoi
from modules.gl.shaders.PosePointLines import PosePointLines

from modules.utils.HotReloadMethods import HotReloadMethods


class CentreCamLayer(LayerBase):
    _blend_shader = Blend()
    _roi_shader = DrawRoi()
    _point_shader = PosePointLines()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, image_renderer: CamImageRenderer,) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._crop_fbo: Fbo = Fbo()
        self._blend_fbo: SwapFbo = SwapFbo()
        self._point_fbo: Fbo = Fbo()
        self._image_renderer: CamImageRenderer = image_renderer
        self._p_pose: Frame | None = None

        self._safe_eye_midpoint: Point2f = Point2f(0.5, 0.5)
        self._safe_height: float = 0.3
        self._safe_rotation: float = 0.0
        self._crop_roi: Rect = Rect(0.0, 0.0, 1.0, 1.0)  # ROI in texture coordinates

        self.data_type: PoseDataHubTypes = data_type
        self.target_x: float = 0.5
        self.target_y: float = 0.25
        self.target_height: float = 1.25
        self.dst_aspectratio: float = 9/16
        self.blend_factor: float = 0.25

        self._on_points_updated = None  # Callback


        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def set_points_callback(self, callback) -> None:
        """Set callback to be called when transformed_points are updated"""
        self._on_points_updated = callback


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._blend_fbo.allocate(width, height, internal_format)
        self._crop_fbo.allocate(width, height, internal_format)
        self._point_fbo.allocate(width, height, internal_format)
        if not CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.allocate(monitor_file=True)
        if not CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.allocate(monitor_file=True)
        if not CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._blend_fbo.deallocate()
        self._crop_fbo.deallocate()
        self._point_fbo.deallocate()
        if CentreCamLayer._blend_shader.allocated:
            CentreCamLayer._blend_shader.deallocate()
        if CentreCamLayer._roi_shader.allocated:
            CentreCamLayer._roi_shader.deallocate()
        if CentreCamLayer._point_shader.allocated:
            CentreCamLayer._point_shader.deallocate()


    def draw(self, rect: Rect) -> None:
        # self._crop_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        self._blend_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        self.draw_points(rect)

        # draw a point at target_x, target_y for debugging (eye midpoint)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glVertex2f(
            rect.x + self.target_x * rect.width,
            rect.y + self.target_y * rect.height
        )
        
        # draw a point at hip midpoint for debugging
        glColor4f(0.0, 1.0, 0.0, 1.0)
        hip_target_y = self.target_y + (self.target_height / 2.0)
        glVertex2f(
            rect.x + self.target_x * rect.width,
            rect.y + hip_target_y * rect.height
        )
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

        key: int = self._cam_id

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._blend_fbo.width, self._blend_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._crop_fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
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

            # Calculate rotation angle from eye to hip midpoint
            delta: Point2f = hip_midpoint - self._safe_eye_midpoint
            self._safe_rotation = math.atan2(delta.x, delta.y)  # Note: x, y order for vertical reference

        self._crop_roi = self._calculate_crop_roi(pose.bbox.to_rect(), self._safe_eye_midpoint, self._safe_height)

        # Convert eye position from bbox-relative to texture-absolute coordinates
        bbox_texture = pose.bbox.to_rect()
        eye_texture_coords = Point2f(
            self._safe_eye_midpoint.x * bbox_texture.width + bbox_texture.x,
            self._safe_eye_midpoint.y * bbox_texture.height + bbox_texture.y
        )

        CentreCamLayer._roi_shader.use(
            self._crop_fbo.fbo_id,
            self._image_renderer._image.tex_id,
            self._crop_roi,
            self._safe_rotation,
            eye_texture_coords,
            False,
            True
        )

        self._blend_fbo.swap()
        CentreCamLayer._blend_shader.use(self._blend_fbo.fbo_id, self._blend_fbo.back_tex_id, self._crop_fbo.tex_id, self.blend_factor)



        # Calculate actual texture aspect ratio
        texture_aspect = self._image_renderer._image.width / self._image_renderer._image.height

        self._transformed_points = self.transform_points_to_crop_space(
            pose.points,
            bbox_texture,
            self._safe_eye_midpoint,
            self._safe_rotation,
            self._crop_roi,
            texture_aspect  # Use actual texture aspect, not dst_aspectratio
        )

        if self._on_points_updated is not None:
            self._on_points_updated(self._transformed_points)
        # return

        line_width: float = 1.0 / self._point_fbo.height * 50
        line_smooth: float = 1.0 / self._point_fbo.height * 25


        self._point_fbo.clear(0.0, 0.0, 0.0, 0.0)
        CentreCamLayer._point_shader.use(self._point_fbo.fbo_id, self._transformed_points , line_width=line_width, line_smooth=line_smooth)


    def _calculate_crop_roi(self, bbox_texture: Rect, eye_bbox_relative: Point2f, height_bbox_relative: float) -> Rect:
        """Calculate the crop ROI in texture coordinates [0,1], centered on the eye position."""
        eye_texture_coords: Point2f = eye_bbox_relative * bbox_texture.size + bbox_texture.position
        height_texture = height_bbox_relative * bbox_texture.height
        width_texture: float = height_texture * self.dst_aspectratio

        return Rect(
            x=eye_texture_coords.x - width_texture * self.target_x,
            y=eye_texture_coords.y - height_texture * self.target_y,
            width=width_texture,
            height=height_texture
        )


    @staticmethod
    def transform_points_to_crop_space(points: Points2D, bbox_texture: Rect, safe_eye_midpoint: Point2f,
                                       safe_rotation: float, crop_roi: Rect, texture_aspect: float) -> Points2D:
        """
        Transform all pose points to crop-space coordinates [0,1].
        Applies the same rotation and translation as the shader.

        Args:
            points: Points2D in bbox-relative coordinates [0,1]
            bbox_texture: The pose bbox in texture coordinates [0,1]
            safe_eye_midpoint: Eye midpoint in bbox-relative coordinates [0,1]
            safe_rotation: Rotation angle in radians
            crop_roi: The crop region in texture coordinates [0,1]
            texture_aspect: Texture aspect ratio (width / height)

        Returns:
            New Points2D in crop ROI space [0,1], accounting for rotation
        """
        # Get rotation center (eye position) in texture coordinates
        eye_texture_coords = Point2f(
            safe_eye_midpoint.x * bbox_texture.width + bbox_texture.x,
            safe_eye_midpoint.y * bbox_texture.height + bbox_texture.y
        )

        # Get x, y arrays from points
        x_bbox, y_bbox = points.get_xy_arrays()

        # 1. Convert from bbox-relative to texture-absolute coordinates
        x_texture = x_bbox * bbox_texture.width + bbox_texture.x
        y_texture = y_bbox * bbox_texture.height + bbox_texture.y

        # 2. Translate to rotation center
        offset_x = x_texture - eye_texture_coords.x
        offset_y = y_texture - eye_texture_coords.y

        # 3. Apply aspect ratio correction
        offset_x *= texture_aspect

        # 4. Apply rotation (same as shader)
        cos_a = np.cos(safe_rotation)
        sin_a = np.sin(safe_rotation)
        rotated_x = cos_a * offset_x - sin_a * offset_y
        rotated_y = sin_a * offset_x + cos_a * offset_y

        # 5. Remove aspect ratio correction
        rotated_x /= texture_aspect

        # 6. Translate back from rotation center
        x_rotated_texture = rotated_x + eye_texture_coords.x
        y_rotated_texture = rotated_y + eye_texture_coords.y

        # 7. Convert from texture coordinates to crop ROI space [0,1]
        x_crop = (x_rotated_texture - crop_roi.x) / crop_roi.width
        y_crop = (y_rotated_texture - crop_roi.y) / crop_roi.height

        # Create new Points2D with transformed coordinates
        return Points2D.from_xy_arrays(x_crop, y_crop, points.scores)