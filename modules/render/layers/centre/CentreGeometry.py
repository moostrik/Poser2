"""Calculates anchor points and geometry for pose-centered rendering."""

# Standard library imports
import math
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D, PointLandmark
from modules.render.layers.LayerBase import LayerBase
from modules.utils.PointsAndRects import Rect, Point2f
from modules.gl import Texture


class CentreGeometry(LayerBase):
    """Calculates anchor points (shoulder/hip midpoints) and derived geometry.

    This layer performs no rendering but computes shared geometry used by
    CentreCamLayer, CentreMaskLayer, and CentrePoseLayer. Also transforms
    pose points to crop space.
    """

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, cam_aspect: float) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._data_type: PoseDataHubTypes = data_type
        self._cam_aspect: float = cam_aspect
        self._p_pose: Frame | None = None

        # Anchor points in bbox-relative coordinates [0,1]
        self._shoulder_midpoint: Point2f = Point2f(0.5, 0.3)
        self._hip_midpoint: Point2f = Point2f(0.5, 0.6)

        # Anchor points in texture coordinates (shared)
        self._anchor_top_tex: Point2f = Point2f(0.5, 0.3)
        self._anchor_bot_tex: Point2f = Point2f(0.5, 0.6)
        self._distance: float = 0.3
        self._bbox: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        # Camera-space geometry (texture coordinates)
        self._cam_rotation: float = 0.0
        self._cam_crop_roi: Rect = Rect(0.0, 0.0, 1.0, 1.0)

        # Bbox-space geometry (for mask and other bbox images)
        self._bbox_aspect: float = 1.0
        self._bbox_rotation: float = 0.0
        self._bbox_crop_roi: Rect = Rect(0.0, 0.0, 1.0, 1.0)
        self._bbox_rotation_center: Point2f = Point2f(0.5, 0.5)

        # Transformed points
        self._transformed_points: Points2D | None = None

        # Configuration
        self.target_top: Point2f = Point2f(0.5, 0.33)
        self.target_bottom: Point2f = Point2f(0.5, 0.6)
        self.dst_aspectratio: float = 9/16

    @property
    def has_pose(self) -> bool:
        """Whether valid pose data exists."""
        return self._p_pose is not None

    @property
    def shoulder_midpoint(self) -> Point2f:
        """Shoulder midpoint in bbox-relative coordinates [0,1]."""
        return self._shoulder_midpoint

    @property
    def hip_midpoint(self) -> Point2f:
        """Hip midpoint in bbox-relative coordinates [0,1]."""
        return self._hip_midpoint

    @property
    def anchor_top_tex(self) -> Point2f:
        """Top anchor point in texture coordinates."""
        return self._anchor_top_tex

    @property
    def anchor_bot_tex(self) -> Point2f:
        """Bottom anchor point in texture coordinates."""
        return self._anchor_bot_tex

    @property
    def cam_rotation(self) -> float:
        """Camera rotation angle in radians."""
        return self._cam_rotation

    @property
    def distance(self) -> float:
        """Distance between anchor points."""
        return self._distance

    @property
    def cam_crop_roi(self) -> Rect:
        """Camera crop region of interest."""
        return self._cam_crop_roi


    @property
    def bbox(self) -> Rect:
        """Current pose bounding box."""
        return self._bbox

    @property
    def bbox_aspect(self) -> float:
        """Bbox aspect ratio (width / height)."""
        return self._bbox_aspect

    @property
    def bbox_rotation(self) -> float:
        """Bbox-space rotation angle in radians (aspect-corrected)."""
        return self._bbox_rotation

    @property
    def bbox_crop_roi(self) -> Rect:
        """Bbox-space crop region of interest."""
        return self._bbox_crop_roi

    @property
    def bbox_rotation_center(self) -> Point2f:
        """Bbox-space rotation center (flipped Y for mask coordinates)."""
        return self._bbox_rotation_center

    @property
    def texture(self) -> Texture:
        """CentreGeometry does not produce a texture output."""
        raise NotImplementedError("CentreGeometry is a computation-only layer with no texture output")

    @property
    def transformed_points(self) -> Points2D | None:
        """Transformed pose points in crop space [0,1], or None if no pose."""
        return self._transformed_points

    def deallocate(self) -> None:
        """No resources to deallocate."""
        pass

    def update(self) -> None:
        """Calculate anchor points and derived geometry from current pose."""
        # Get pose data
        pose: Frame | None = self._data_hub.get_item(DataHubType(self._data_type), self._cam_id)
        if pose is self._p_pose:
            return
        self._p_pose = pose

        if pose is None:
            self._transformed_points = None
            return

        # Update body landmarks (bbox-relative coordinates)
        shoulder_mid = CentreGeometry._get_midpoint(
            pose.points, PointLandmark.left_shoulder, PointLandmark.right_shoulder
        )
        if shoulder_mid:
            self._shoulder_midpoint = shoulder_mid

        hip_mid = CentreGeometry._get_midpoint(
            pose.points, PointLandmark.left_hip, PointLandmark.right_hip
        )
        if hip_mid:
            self._hip_midpoint = hip_mid

        # Convert to texture coordinates
        self._bbox = pose.bbox.to_rect()
        self._anchor_top_tex = self._shoulder_midpoint * self._bbox.size + self._bbox.position
        self._anchor_bot_tex = self._hip_midpoint * self._bbox.size + self._bbox.position

        # Calculate camera-space geometry
        target_distance: float = self.target_bottom.y - self.target_top.y
        self._cam_rotation, self._distance, self._cam_crop_roi = CentreGeometry._calculate_roi(
            self._anchor_top_tex, self._anchor_bot_tex, self.target_top, target_distance, self.dst_aspectratio
        )

        # Calculate bbox-space geometry (aspect-corrected for mask)
        self._bbox_aspect = self._bbox.width / self._bbox.height if self._bbox.height > 0 else 1.0
        mask_delta = self._hip_midpoint - self._shoulder_midpoint
        # Negate rotation for flipped bbox coordinate system
        self._bbox_rotation = -math.atan2(mask_delta.x * self._bbox_aspect, mask_delta.y)
        mask_distance = math.hypot(mask_delta.x * self._bbox_aspect, mask_delta.y)

        mask_height = mask_distance / target_distance
        mask_width = mask_height * self._bbox_aspect

        self._bbox_crop_roi = Rect(
            x=self._shoulder_midpoint.x - mask_width * self.target_top.x,
            y=1.0 - self._shoulder_midpoint.y - mask_height * (1.0 - self.target_top.y),
            width=mask_width,
            height=mask_height
        )

        self._bbox_rotation_center = Point2f(
            self._shoulder_midpoint.x,
            1.0 - self._shoulder_midpoint.y
        )

        # Transform pose points using camera geometry
        self._transformed_points = CentreGeometry._transform_points(
            pose.points, self._bbox, self._anchor_top_tex, self._cam_aspect, self._cam_rotation, self._cam_crop_roi
        )

    def draw(self, rect: Rect) -> None:
        """No rendering performed by this layer."""
        pass

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
