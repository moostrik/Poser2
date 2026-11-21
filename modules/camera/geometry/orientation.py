"""The camera's orientation from its IMU: the gravity vector into the camera's frame, and its tilt and roll."""

import math
from typing import cast

from ..settings import RigCameraReadings

IMU_RATE_HZ: int = 5
IMU_SMOOTHING: float = 0.1      # how hard each sample pulls the average: the camera is on a tripod, a reading is mostly noise


class MountReadout:
    """The mount readout: the accelerometer's samples into the camera's frame and smoothed into one gravity vector,
    its tilt and its roll less `roll_offset` into the readings. `board_roll` is the IMU's turn on the board (°);
    `imu_rotation` the board's IMU-to-camera extrinsics, None where it has none."""

    def __init__(self, readings: RigCameraReadings, board_roll: float) -> None:
        self.readings: RigCameraReadings = readings
        self.board_roll: float = board_roll
        self.imu_rotation: list[list[float]] | None = None
        self.gravity: tuple[float, float, float] | None = None

    def update(self, samples: list[tuple[float, float, float]]) -> float | None:
        """The samples in, the readings set; the tilt (°), None before the first sample."""
        for sample in samples:
            vector: tuple[float, float, float] = unroll_imu_frame(imu_to_camera(sample, self.imu_rotation),
                                                                  self.board_roll)
            if self.gravity is None:
                self.gravity = vector
            else:
                self.gravity = cast(tuple[float, float, float], tuple(
                    previous + IMU_SMOOTHING * (current - previous)
                    for previous, current in zip(self.gravity, vector)
                ))
        if self.gravity is None:
            return None
        tilt, roll = orientation_from_gravity(*self.gravity)
        # Tilt gets no offset: there is no way to measure true tilt on site, so there would be nothing to calibrate
        # it against.
        self.readings.tilt_measured = tilt
        self.readings.roll_measured = roll - self.readings.roll_offset
        return tilt


def orientation_from_gravity(gx: float, gy: float, gz: float) -> tuple[float, float]:
    """Tilt and roll (°) from the gravity vector in the camera's frame (x right, y down, z forward); tilt
    positive aimed up, roll positive toward +x, NaN for a vector without length."""
    magnitude: float = math.sqrt(gx * gx + gy * gy + gz * gz)
    if magnitude < 1e-6:
        return (float('nan'), float('nan'))
    tilt: float = math.degrees(math.atan2(-gz, math.hypot(gx, gy)))
    roll: float = math.degrees(math.atan2(gx, gy))
    return (tilt, roll)


def imu_to_camera(vector: tuple[float, float, float],
                  extrinsics: list[list[float]] | None) -> tuple[float, float, float]:
    """A vector from the IMU's frame into the camera's, by the rotation of the 4x4 `extrinsics`; the vector as
    it is without them."""
    if not extrinsics or len(extrinsics) < 3:
        return vector
    try:
        return cast(tuple[float, float, float], tuple(
            sum(extrinsics[row][col] * vector[col] for col in range(3))
            for row in range(3)
        ))
    except (IndexError, TypeError):
        return vector


def unroll_imu_frame(vector: tuple[float, float, float], board_roll: float) -> tuple[float, float, float]:
    """The vector with the IMU's mounting turn about the optical axis, `board_roll` (°), undone."""
    angle: float = math.radians(board_roll)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)
