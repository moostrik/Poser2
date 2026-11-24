"""Calculates anchor points and geometry for pose-centered rendering."""

# Standard library imports
import math
import numpy as np
from enum import IntEnum
from dataclasses import dataclass

# Local application imports
from modules.board import HasFrames
from modules.flow import fixed_window
from modules.pose import Frame, Points2D, PointLandmark, BBox
from modules.render import LayerBase, DataCache
from modules.utils import Rect, Point2f, Quad
from modules.gl import Texture
from ....settings import CenterGeometrySettings, SnapPoint


class SourceSpace(IntEnum):
    """The space a centered layer's source texture is in."""
    CROP  = 0   # the pose's crop (a model's output on it, as RVM's mask): `bbox_geometry`
    IMAGE = 1   # the camera image (the image itself, depth on its view): `image_geometry`


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


class CenterGeometry(LayerBase):
    """Calculates anchor points (shoulder/hip midpoints) and derived geometry.

    This layer performs no rendering but computes shared geometry used by
    CenterCamLayer, CenterMaskLayer, and ViewPoseLayer. Also transforms
    pose points to crop space. Image coordinates are normalized per axis, so `cam_texture`'s aspect
    is what turns them into true proportions.
    """

    def __init__(self, cam_id: int, board: HasFrames, cam_texture: Texture, settings: CenterGeometrySettings | None = None, fov: float = 0.0) -> None:
        """`fov` is the camera image's horizontal field of view (degrees), for `view_span`; 0 for a camera without depth."""
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._cam_texture: Texture = cam_texture
        self._image_span: float = 2.0 * math.tan(math.radians(fov) / 2.0)     # metres the camera image spans per metre of depth
        self._data_cache: DataCache[Frame] = DataCache[Frame]()

        # Configuration
        self.settings: CenterGeometrySettings = settings or CenterGeometrySettings()

        # Geometry results
        self._cam_geometry: CamGeometry = CamGeometry(
            Rect(0.0, 0.0, 1.0, 1.0), 0.0, Point2f(0.5, 0.3)
        )
        self._bbox_geometry: BboxGeometry = BboxGeometry(
            Rect(0.0, 0.0, 1.0, 1.0), 0.0, Point2f(0.5, 0.5), 1.0
        )
        self._transformed_points: Points2D | None = None

        # Last valid pin midpoint (bbox-relative), held when the selected
        # snap point is momentarily lost (e.g. feet out of frame).
        self._last_pin_mid: Point2f | None = None
        self._last_snap_point: SnapPoint | None = None

    @property
    def present(self) -> bool:
        """Whether valid pose data exists."""
        return self._data_cache.has_data

    @property
    def lost(self) -> bool:
        """Whether the view was lost since last update: the pose went, while the view follows it."""
        return self.settings.follow_pose and self._data_cache.lost

    @property
    def ready(self) -> bool:
        """Whether the view is defined: placed by a pose while following it, always when fixed."""
        if self.settings.follow_pose:
            return self._transformed_points is not None
        return self._cam_texture.allocated

    @property
    def image_geometry(self) -> CamGeometry:
        """Camera rendering geometry (crop ROI, rotation, rotation center)."""
        return self._cam_geometry

    @property
    def bbox_geometry(self) -> BboxGeometry:
        """Bbox rendering geometry for mask/flow (crop ROI, rotation, rotation center, aspect)."""
        return self._bbox_geometry

    @property
    def view_span(self) -> float:
        """The metres one width of the view spans per metre of depth: the crop's share of the camera
        image's field of view; 0 without a field of view."""
        return self._cam_geometry.crop_roi.width * self._image_span

    def depth_step(self, sample_size: tuple[float, float]) -> tuple[float, float]:
        """The uv between two samples in the view, for a texture on the camera image whose samples are
        `sample_size` (uv) apart; (0, 0) while the view has no size."""
        crop: Rect = self._cam_geometry.crop_roi
        if crop.width <= 0.0 or crop.height <= 0.0:
            return (0.0, 0.0)
        return (sample_size[0] / crop.width, sample_size[1] / crop.height)

    def source_quad(self, space: SourceSpace, source: Texture) -> Quad:
        """The part of a texture in `space` that centers it, as `texture.draw`'s `src`: its ROI turned about the
        rotation center, round in the source's pixels. The camera image's aspect is the source texture's own."""
        if space == SourceSpace.CROP:
            bbox: BboxGeometry = self._bbox_geometry
            return bbox.crop_roi.turned(bbox.rotation, bbox.rotation_center, bbox.aspect)
        image: CamGeometry = self._cam_geometry
        aspect: float = source.width / source.height if source.height > 0 else 1.0
        return image.crop_roi.turned(image.rotation, image.rotation_center, aspect)

    @property
    def crop_pose_points(self) -> Points2D | None:
        """Pose points in crop space [0,1], or None if no pose."""
        return self._transformed_points

    @property
    def texture(self) -> Texture:
        """CenterGeometry does not produce a texture output."""
        raise NotImplementedError("CenterGeometry is a computation-only layer with no texture output")

    def deallocate(self) -> None:
        """No resources to deallocate."""
        pass

    def update(self) -> None:
        """Calculate anchor points and derived geometry from current pose."""
        # Get pose data
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._cam_id)
        self._data_cache.set(pose)
        if not self.settings.follow_pose:
            self._update_fixed(pose)
            return
        if self._data_cache.idle:
            return

        if pose is None:
            self._transformed_points = None
            self._last_pin_mid = None
            return

        # Reconstruct the target (where the pinned point lands) from settings
        target = Point2f(self.settings.target_x, self.settings.target_y)

        # Get current body landmarks (bbox-relative coordinates)
        shoulder_mid = CenterGeometry._get_midpoint(
            pose[Points2D], PointLandmark.left_shoulder, PointLandmark.right_shoulder
        )
        hip_mid = CenterGeometry._get_midpoint(
            pose[Points2D], PointLandmark.left_hip, PointLandmark.right_hip
        )

        if not shoulder_mid or not hip_mid:
            self._transformed_points = None
            return

        # Select the pinned point (bbox-relative). Scale & rotation always come from
        # the torso (shoulder-hip); only the pinned position changes here.
        feet_mid = CenterGeometry._get_midpoint(
            pose[Points2D], PointLandmark.left_ankle, PointLandmark.right_ankle
        )

        # Raw midpoint for the selected snap point (feet may be None this frame)
        if self.settings.snap_point == SnapPoint.HIPS:
            raw_pin = hip_mid
        elif self.settings.snap_point == SnapPoint.FEET:
            raw_pin = feet_mid
        else:
            raw_pin = shoulder_mid

        # Reset the held value when the user switches snap point
        if self.settings.snap_point != self._last_snap_point:
            self._last_pin_mid = None
            self._last_snap_point = self.settings.snap_point

        if raw_pin is not None:
            pin_mid = raw_pin
            self._last_pin_mid = raw_pin          # remember last good position
        elif self._last_pin_mid is not None:
            pin_mid = self._last_pin_mid          # hold last value (e.g. feet lost)
        else:
            pin_mid = hip_mid                     # cold start: no history yet

        # Calculate anchor points in image space (full image coordinates)
        bbox = pose[BBox].to_rect()
        anchor_top_img = shoulder_mid * bbox.size + bbox.top_left
        anchor_bot_img = hip_mid * bbox.size + bbox.top_left
        pin_img = pin_mid * bbox.size + bbox.top_left

        # Calculate camera-space geometry. Guard against a degenerate size
        # which would otherwise divide by zero when scaling.
        target_distance: float = self.settings.target_size
        if abs(target_distance) < 1e-6:
            self._transformed_points = None
            return
        cam_height: int = self._cam_texture.height
        cam_aspect: float = self._cam_texture.width / cam_height if cam_height > 0 else 1.0
        cam_rotation_img, distance, cam_crop_roi_img = CenterGeometry._calculate_roi(
            anchor_top_img, anchor_bot_img, pin_img, target, target_distance, self.settings.dst_aspectratio, cam_aspect
        )
        if not self.settings.torso_upright:
            cam_rotation_img = 0.0

        self._cam_geometry = CamGeometry(crop_roi=cam_crop_roi_img, rotation=cam_rotation_img, rotation_center=pin_img)

        # Calculate bbox-space geometry (aspect-corrected for mask)
        bbox_aspect = bbox.width * cam_aspect / bbox.height if bbox.height > 0 else 1.0
        mask_delta = hip_mid - shoulder_mid
        mask_rotation = math.atan2(mask_delta.x * bbox_aspect, mask_delta.y) if self.settings.torso_upright else 0.0
        mask_distance = math.hypot(mask_delta.x * bbox_aspect, mask_delta.y)

        mask_height = mask_distance / target_distance
        mask_width = mask_height * bbox_aspect

        # Store bbox geometry (positioned around the pinned point)
        self._bbox_geometry = BboxGeometry(
            crop_roi=Rect(
                x=pin_mid.x - mask_width * target.x,
                y=pin_mid.y - mask_height * target.y,
                width=mask_width,
                height=mask_height
            ),
            rotation=mask_rotation,
            rotation_center=pin_mid,
            aspect=bbox_aspect
        )

        # Transform pose points using camera geometry (in image space)
        self._transformed_points = CenterGeometry._transform_points(
            pose[Points2D], bbox, pin_img, cam_aspect, cam_rotation_img, cam_crop_roi_img
        )

    def _update_fixed(self, pose: Frame | None) -> None:
        """The fixed window of the camera image, whatever the pose; the pose only places its points in it,
        and the window in its crop for a source on the crop."""
        cam_height: int = self._cam_texture.height
        if cam_height <= 0:
            return
        cam_aspect: float = self._cam_texture.width / cam_height
        s: CenterGeometrySettings = self.settings
        window: Rect = fixed_window(s.fixed_x, s.fixed_y, s.fixed_height, s.dst_aspectratio, cam_aspect)
        center: Point2f = Point2f(s.fixed_x, s.fixed_y)

        self._cam_geometry = CamGeometry(crop_roi=window, rotation=0.0, rotation_center=center)

        if pose is None:
            self._transformed_points = None
            return
        bbox: Rect = pose[BBox].to_rect()
        if bbox.width > 0 and bbox.height > 0:
            # The same window in the pose's crop, for a source on the crop (RVM's mask)
            x: float = (window.x - bbox.x) / bbox.width
            y: float = (window.y - bbox.y) / bbox.height
            width: float = window.width / bbox.width
            height: float = window.height / bbox.height
            self._bbox_geometry = BboxGeometry(
                crop_roi=Rect(x, y, width, height),
                rotation=0.0,
                rotation_center=Point2f(x + width / 2.0, y + height / 2.0),
                aspect=bbox.width * cam_aspect / bbox.height
            )
        self._transformed_points = CenterGeometry._transform_points(pose[Points2D], bbox, center, cam_aspect, 0.0, window)

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
    def _calculate_roi(scale_top: Point2f, scale_bot: Point2f, pin_point: Point2f,
                      target: Point2f, target_distance: float,
                      output_aspect: float, cam_aspect: float = 1.0) -> tuple[float, float, Rect]:
        """Calculate rotation angle and ROI for cropping.

        Rotation and scale derive from the torso reference (scale_top -> scale_bot),
        while the ROI is positioned so that pin_point lands at target. The points and the
        ROI are normalized per axis; `cam_aspect` is the frame's width over its height.

        Returns: (rotation_angle, distance, roi_rect)
        """
        delta = scale_bot - scale_top
        rotation = math.atan2(delta.x * cam_aspect, delta.y)
        distance = math.hypot(delta.x * cam_aspect, delta.y)

        height = distance / target_distance
        width = height * output_aspect / cam_aspect

        roi = Rect(
            x=pin_point.x - width * target.x,
            y=pin_point.y - height * target.y,
            width=width,
            height=height
        )

        return rotation, distance, roi

    @staticmethod
    def _transform_points(points: Points2D, bbox: Rect, top: Point2f,
                          aspect: float, rotation: float, crop_roi: Rect) -> Points2D:
        """Transform pose points to crop-space coordinates [0,1]."""
        x_bbox, y_bbox = points.get_xy_arrays()

        # Convert from bbox-relative to image coordinates
        x = x_bbox * bbox.width + bbox.x
        y = y_bbox * bbox.height + bbox.y

        # Rotate around shoulder position, in the frame's true proportions
        dx = (x - top.x) * aspect
        dy = y - top.y

        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        x_rot = (cos_a * dx - sin_a * dy) / aspect + top.x
        y_rot = sin_a * dx + cos_a * dy + top.y

        # Convert to crop ROI space
        x_crop = (x_rot - crop_roi.x) / crop_roi.width
        y_crop = (y_rot - crop_roi.y) / crop_roi.height

        return Points2D.from_xy_arrays(x_crop, y_crop, points.scores)
