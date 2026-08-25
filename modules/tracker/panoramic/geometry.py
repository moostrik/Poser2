import math
from enum import IntEnum

import numpy as np

from modules.utils import Rect


# Distance estimation is clamped to a plausible range so a degenerate bounding
# box can never produce a nonsensical parallax correction.
_MIN_DISTANCE: float = 0.5
_MAX_DISTANCE: float = 10.0


class DistortAlgorithm(IntEnum):
    NONE = 0   # identity — no distortion correction
    POLY = 1   # polynomial: x + k1*(x-0.5) + k2*(x-0.5)^3
    TANH = 2   # S-curve:  0.5 * (1 + tanh(k1*(2x-1) + k2*(2x-1)^3))


class Geometry:
    def __init__(self, num_cameras: int, cam_fov: float, target_fov: float) -> None:
        self.num_cameras: int = num_cameras
        self.cam_fov: float = cam_fov
        self.target_fov: float = target_fov
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

        self._tanh_slope: float = 0.0
        self._tanh_cubic: float = 0.0
        self._poly_k1: float = 0.0
        self._poly_k2: float = 0.0
        self.algorithm: DistortAlgorithm = DistortAlgorithm.NONE

        # Parallax: cameras sit on a ring of this radius (m), not at the shared
        # centre the world-angle model assumes. 0 disables the correction.
        self._ring_radius: float = 0.0
        self._person_height: float = 1.7
        self._vfov: float = 71.6

    def get_angles_and_overlap(self, roi: Rect, cam_id: int, expansion: float) -> tuple[float, float, bool, float]:
        local_angle, world_angle, distance = self.calc_angle(roi, cam_id)
        overlap: bool = self.angle_in_overlap(local_angle, expansion)
        return (local_angle, world_angle, overlap, distance)

    def calc_angle(self, roi: Rect, cam_id: int) -> tuple[float, float, float]:
        local_angle: float = self._calc_local_angle(roi)
        distance: float = self.estimate_distance(roi)
        # Edge/overlap/hysteresis tests stay in the raw camera frame; only the
        # world angle is re-projected to the shared centre for cross-camera fusion.
        corrected_local: float = self._parallax_corrected_local(local_angle, distance)
        world_angle: float = self._calc_world_angle(corrected_local, cam_id)
        return local_angle, world_angle, distance

    def estimate_distance(self, roi: Rect) -> float:
        """Estimate distance from the camera (m) from the bounding-box height,
        assuming a person of ``person_height`` filling ``roi.height`` of a frame
        with vertical field of view ``vfov``. Clamped to a plausible range."""
        angular_height: float = roi.height * self._vfov
        if angular_height <= 0.0:
            return _MAX_DISTANCE
        half_angle: float = math.radians(angular_height / 2.0)
        distance: float = (self._person_height / 2.0) / math.tan(half_angle)
        return max(_MIN_DISTANCE, min(_MAX_DISTANCE, distance))

    def _parallax_corrected_local(self, local_angle: float, distance: float) -> float:
        """Re-project a local angle so it reads as if seen from the rig centre.

        The camera faces radially outward, so the centre sits ``ring_radius``
        behind it. Placing the person at (distance, angle) in the camera frame
        and re-measuring the bearing from the centre removes the cross-camera
        seam disagreement caused by the off-centre mounting."""
        if self._ring_radius <= 0.0:
            return local_angle
        theta: float = math.radians(local_angle - self.cam_fov / 2.0)
        x: float = distance * math.cos(theta) + self._ring_radius
        y: float = distance * math.sin(theta)
        return math.degrees(math.atan2(y, x)) + self.cam_fov / 2.0

    def _calc_local_angle(self, roi: Rect) -> float:
        normalized_x: float = roi.x + roi.width / 2.0
        normalized_x = self.undistort_x(normalized_x)
        local_angle: float = normalized_x * self.cam_fov
        return local_angle

    def _calc_world_angle(self, local_angle: float, cam_id: int) -> float:
        world_angle: float = self.target_fov * cam_id + local_angle - self.fov_overlap
        world_angle = world_angle % 360.0  # Ensure the angle is within 0 to 360 degrees
        return world_angle

    def angle_in_overlap(self, local_angle: float, expansion: float = 0.0) -> bool:
        angle_overlap: float = self.fov_overlap * (1.0 + expansion)

        if local_angle <= angle_overlap or local_angle >= self.cam_fov - angle_overlap:
            return True
        return False

    def angle_in_edge(self, local_angle: float, range_: float = 1.0) -> bool:
        edge: float = self.fov_overlap * range_

        if local_angle <= edge or local_angle >= self.cam_fov - edge:
            return True
        return False

    def angle_from_edge(self, local_angle: float) -> float:
        return min(local_angle, self.cam_fov - local_angle)

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        diff: float = abs(a - b)
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def undistort_x(self, x: float) -> float:
        if self.algorithm == DistortAlgorithm.NONE:
            return x
        elif self.algorithm == DistortAlgorithm.TANH:
            return 0.5 * (1.0 + np.tanh(self._tanh_slope * (2*x - 1) + self._tanh_cubic * (2*x - 1)**3))
        else:  # poly
            d = x - 0.5
            return x + self._poly_k1 * d + self._poly_k2 * d**3

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self.fov_overlap = (self.cam_fov - self.target_fov) / 2.0

    def set_tanh_slope(self, slope: float) -> None:
        self._tanh_slope = slope

    def set_tanh_cubic(self, cubic: float) -> None:
        self._tanh_cubic = cubic

    def set_poly_k1(self, k1: float) -> None:
        self._poly_k1 = k1

    def set_poly_k2(self, k2: float) -> None:
        self._poly_k2 = k2

    def set_algorithm(self, algorithm: DistortAlgorithm) -> None:
        self.algorithm = algorithm

    def set_ring_radius(self, ring_radius: float) -> None:
        self._ring_radius = ring_radius

    def set_person_height(self, person_height: float) -> None:
        self._person_height = person_height

    def set_vfov(self, vfov: float) -> None:
        self._vfov = vfov
