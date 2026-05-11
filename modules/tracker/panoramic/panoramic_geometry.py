from enum import IntEnum

import numpy as np

from modules.utils import Rect


class DistortAlgorithm(IntEnum):
    NONE = 0   # identity — no distortion correction
    TANH = 1   # S-curve:  0.5 * (1 + tanh(k1*(2x-1) + k2*(2x-1)^3))
    POLY = 2   # polynomial: x + k1*(x-0.5) + k2*(x-0.5)^3


class PanoramicGeometry:
    def __init__(self, num_cameras: int, cam_fov: float, target_fov: float) -> None:
        self.num_cameras: int = num_cameras
        self.cam_fov: float = cam_fov
        self.target_fov: float = target_fov
        self.fov_overlap: float = (self.cam_fov - self.target_fov) / 2.0

        self.k1: float = 0.0
        self.k2: float = 0.0
        self.algorithm: DistortAlgorithm = DistortAlgorithm.NONE

    def get_angles_and_overlap(self, roi: Rect, cam_id: int, expansion: float) -> tuple[float, float, bool]:
        local_angle, world_angle = self.calc_angle(roi, cam_id)
        overlap: bool = self.angle_in_overlap(world_angle, expansion)
        return (local_angle, world_angle, overlap)

    def calc_angle(self, roi: Rect, cam_id: int) -> tuple[float, float]:
        local_angle: float = self._calc_local_angle(roi)
        world_angle: float = self._calc_world_angle(local_angle, cam_id)
        return local_angle, world_angle

    def _calc_local_angle(self, roi: Rect) -> float:
        normalized_x: float = roi.x + roi.width / 2.0
        normalized_x = self.undistort_x(normalized_x)
        local_angle: float = normalized_x * self.cam_fov
        return local_angle

    def _calc_world_angle(self, local_angle: float, cam_id: int) -> float:
        world_angle: float = self.target_fov * cam_id + local_angle - self.fov_overlap
        world_angle = world_angle % 360.0  # Ensure the angle is within 0 to 360 degrees
        if world_angle < 0:
            world_angle += 360.0
        return world_angle

    def angle_in_overlap(self, world_angle: float, expansion: float = 0.0) -> bool:
        angle_overlap: float = self.fov_overlap * (1.0 + expansion)
        local_angle: float = world_angle % self.target_fov

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
            return 0.5 * (1.0 + np.tanh(self.k1 * (2*x - 1) + self.k2 * (2*x - 1)**3))
        else:  # poly
            d = x - 0.5
            return x + self.k1 * d + self.k2 * d**3

    # SET
    def set_fov(self, cam_fov: float) -> None:
        self.cam_fov = cam_fov
        self.fov_overlap = (self.cam_fov - self.target_fov) / 2.0

    def set_k1(self, k1: float) -> None:
        self.k1 = k1

    def set_k2(self, k2: float) -> None:
        self.k2 = k2

    def set_algorithm(self, algorithm: DistortAlgorithm) -> None:
        self.algorithm = algorithm
