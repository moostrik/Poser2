"""The camera's orientation from its IMU: the gravity vector into the camera's frame, and its tilt and roll."""

import math
from typing import cast


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
