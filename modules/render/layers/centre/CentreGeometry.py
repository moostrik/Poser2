"""Calculates anchor points and geometry for pose-centered rendering."""

# Standard library imports
import math
import numpy as np
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import Points2D, PointLandmark
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.utils.PointsAndRects import Rect, Point2f
from modules.gl import Texture

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class CamGeometry:
    """Camera rendering geometry."""
    crop_roi: Rect
    rotation: float
    rotation_center: Point2f


@dataclass
class BboxGeometry:
    """Bounding box rendering geometry (for mask/flow)."""
    crop_roi: Rect
    rotation: float
    rotation_center: Point2f
    aspect: float


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
        self._data_cache: DataCache[Frame] = DataCache[Frame]()

        # Geometry results
        self._cam_geometry: CamGeometry = CamGeometry(
            Rect(0.0, 0.0, 1.0, 1.0), 0.0, Point2f(0.5, 0.3)
        )
        self._bbox_geometry: BboxGeometry = BboxGeometry(
            Rect(0.0, 0.0, 1.0, 1.0), 0.0, Point2f(0.5, 0.5), 1.0
        )
        self._transformed_points: Points2D | None = None

        # Configuration
        self.target_top: Point2f = Point2f(0.5, 0.33)
        self.target_bottom: Point2f = Point2f(0.5, 0.6)
        self.dst_aspectratio: float = 9/16

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def present(self) -> bool:
        """Whether valid pose data exists."""
        return self._data_cache.has_data

    @property
    def lost(self) -> bool:
        """Whether pose data was lost since last update."""
        return self._data_cache.lost

    @property
    def image_geometry(self) -> CamGeometry:
        """Camera rendering geometry (crop ROI, rotation, rotation center)."""
        return self._cam_geometry

    @property
    def bbox_geometry(self) -> BboxGeometry:
        """Bbox rendering geometry for mask/flow (crop ROI, rotation, rotation center, aspect)."""
        return self._bbox_geometry

    @property
    def crop_pose_points(self) -> Points2D | None:
        """Pose points in crop space [0,1], or None if no pose."""
        return self._transformed_points

    @property
    def texture(self) -> Texture:
        """CentreGeometry does not produce a texture output."""
        raise NotImplementedError("CentreGeometry is a computation-only layer with no texture output")

    def deallocate(self) -> None:
        """No resources to deallocate."""
        pass

    def update(self) -> None:
        """Calculate anchor points and derived geometry from current pose."""
        # Get pose data
        pose: Frame | None = self._data_hub.get_item(DataHubType(self._data_type), self._cam_id)
        self._data_cache.update(pose)
        if self._data_cache.idle:
            return

        if pose is None:
            self._transformed_points = None
            return

        # Get current body landmarks (bbox-relative coordinates)
        shoulder_mid = CentreGeometry._get_midpoint(
            pose.points, PointLandmark.left_shoulder, PointLandmark.right_shoulder
        )
        hip_mid = CentreGeometry._get_midpoint(
            pose.points, PointLandmark.left_hip, PointLandmark.right_hip
        )

        if not shoulder_mid or not hip_mid:
            self._transformed_points = None
            return

        # Calculate anchor points in image space (full image coordinates)
        bbox = pose.bbox.to_rect()
        anchor_top_img = shoulder_mid * bbox.size + bbox.position
        anchor_bot_img = hip_mid * bbox.size + bbox.position

        # Convert anchor points to texture space (bottom-left origin)
        anchor_top_tex = CentreGeometry._image_to_texture_point(anchor_top_img)

        # Calculate camera-space geometry
        target_distance: float = self.target_bottom.y - self.target_top.y
        cam_rotation_img, distance, cam_crop_roi_img = CentreGeometry._calculate_roi(
            anchor_top_img, anchor_bot_img, self.target_top, target_distance, self.dst_aspectratio
        )

        # Store camera geometry (texture space)
        self._cam_geometry = CamGeometry(
            crop_roi=Rect(
                cam_crop_roi_img.x,
                CentreGeometry._image_to_texture_y(cam_crop_roi_img.y + cam_crop_roi_img.height),
                cam_crop_roi_img.width,
                cam_crop_roi_img.height
            ),
            rotation=CentreGeometry._image_to_texture_rotation(cam_rotation_img),
            rotation_center=anchor_top_tex
        )

        # Calculate bbox-space geometry (aspect-corrected for mask)
        bbox_aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0
        mask_delta = hip_mid - shoulder_mid
        mask_rotation = -math.atan2(mask_delta.x * bbox_aspect, mask_delta.y)
        mask_distance = math.hypot(mask_delta.x * bbox_aspect, mask_delta.y)

        mask_height = mask_distance / target_distance
        mask_width = mask_height * bbox_aspect

        # Store bbox geometry
        self._bbox_geometry = BboxGeometry(
            crop_roi=Rect(
                x=shoulder_mid.x - mask_width * self.target_top.x,
                y=1.0 - shoulder_mid.y - mask_height * (1.0 - self.target_top.y),
                width=mask_width,
                height=mask_height
            ),
            rotation=mask_rotation,
            rotation_center=Point2f(shoulder_mid.x, 1.0 - shoulder_mid.y),
            aspect=bbox_aspect
        )

        # Transform pose points using camera geometry (in image space)
        self._transformed_points = CentreGeometry._transform_points(
            pose.points, bbox, anchor_top_img, self._cam_aspect, cam_rotation_img, cam_crop_roi_img
        )

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
    def _image_to_texture_y(y: float) -> float:
        """Convert Y coordinate from image space (top-left) to texture space (bottom-left)."""
        return 1.0 - y

    @staticmethod
    def _image_to_texture_point(point: Point2f) -> Point2f:
        """Convert point from image space to texture space."""
        return Point2f(point.x, 1.0 - point.y)

    @staticmethod
    def _image_to_texture_rotation(rotation: float) -> float:
        """Convert rotation from image space to texture space (negate)."""
        return -rotation

    @staticmethod
    def _transform_points(points: Points2D, bbox: Rect, top: Point2f,
                          aspect: float, rotation: float, crop_roi: Rect) -> Points2D:
        """Transform pose points to crop-space coordinates [0,1]."""
        x_bbox, y_bbox = points.get_xy_arrays()

        # Convert from bbox-relative to image coordinates
        x = x_bbox * bbox.width + bbox.x
        y = y_bbox * bbox.height + bbox.y

        # Rotate around shoulder position (no aspect correction in image space)
        dx = x - top.x
        dy = y - top.y

        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        x_rot = cos_a * dx - sin_a * dy + top.x
        y_rot = sin_a * dx + cos_a * dy + top.y

        # Convert to crop ROI space
        x_crop = (x_rot - crop_roi.x) / crop_roi.width
        y_crop = (y_rot - crop_roi.y) / crop_roi.height

        return Points2D.from_xy_arrays(x_crop, y_crop, points.scores)
