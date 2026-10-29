"""The OAK board's IMU frame: how a gravity vector gets into the camera's frame, and the quarter
turn the board mounts it at (`IMU_BOARD_ROLL`). The angles themselves are the contract's
(`modules/camera/tests/test_orientation.py`)."""

import math
import unittest

from modules.camera import orientation_from_gravity, imu_to_camera, unroll_imu_frame
from modules.camera.oak.definitions import IMU_BOARD_ROLL


def gravity(tilt: float, roll: float) -> tuple[float, float, float]:
    """The gravity vector a camera at this tilt and roll would see, in its own frame.

    Built forwards from the rotations rather than from the formula under test: tilt about x,
    then roll about z, applied to the level reading (0, 1, 0).
    """
    t, r = math.radians(tilt), math.radians(roll)
    x, y, z = 0.0, 1.0, 0.0
    y, z = y * math.cos(t), -y * math.sin(t)            # pitch up about x
    x, y = x * math.cos(r) + y * math.sin(r), y * math.cos(r) - x * math.sin(r)
    return (x, y, z)


class TestImuToCamera(unittest.TestCase):
    def test_identity_and_missing_pass_through(self) -> None:
        v = (0.1, 0.9, -0.2)
        identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        self.assertEqual(imu_to_camera(v, identity), v)
        self.assertEqual(imu_to_camera(v, None), v)
        self.assertEqual(imu_to_camera(v, []), v)

    def test_a_known_rotation(self) -> None:
        """90 degrees about z: x <- -y, y <- x."""
        quarter = [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        x, y, z = imu_to_camera((0.0, 1.0, 0.0), quarter)
        self.assertAlmostEqual(x, -1.0, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertAlmostEqual(z, 0.0, places=9)


def as_imu(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """What the board actually reports for a camera-frame gravity vector.

    The IMU sits a quarter turn about the optical axis, so its reading is `Rz(+90)` of the camera
    frame — this is the misalignment `unroll_imu_frame` undoes. Verified on the rig: all four
    cameras read roll -90 while their tilt read true.
    """
    x, y, z = vector
    return (-y, x, z)


class TestUnrollImuFrame(unittest.TestCase):
    def test_a_level_camera_reads_minus_ninety_before_correction(self) -> None:
        """The observation this constant was derived from."""
        _, roll = orientation_from_gravity(*as_imu(gravity(0.0, 0.0)))
        self.assertAlmostEqual(roll, -90.0, places=6)

    def test_tilt_survives_the_misalignment_uncorrected(self) -> None:
        """Why the tilt readout was trustworthy before the correction existed: a rotation about z
        leaves gz alone and hypot(gx, gy) invariant."""
        for angle in (0.0, 12.0, 16.0, 25.0):
            tilt, _ = orientation_from_gravity(*as_imu(gravity(angle, 0.0)))
            self.assertAlmostEqual(tilt, angle, places=6, msg=f'tilt {angle}')

    def test_correction_recovers_the_camera_frame(self) -> None:
        for tilt_in in (0.0, 12.0, 16.0):
            for roll_in in (-3.0, 0.0, 3.0, 7.5):
                tilt, roll = orientation_from_gravity(*unroll_imu_frame(as_imu(gravity(tilt_in, roll_in)), IMU_BOARD_ROLL))
                self.assertAlmostEqual(tilt, tilt_in, places=6, msg=f'{tilt_in}/{roll_in}')
                self.assertAlmostEqual(roll, roll_in, places=6, msg=f'{tilt_in}/{roll_in}')

    def test_a_real_tripod_roll_is_not_swallowed(self) -> None:
        """The point of the correction: what is left after the quarter turn is the actual roll."""
        _, roll = orientation_from_gravity(*unroll_imu_frame(as_imu(gravity(12.0, 4.0)), IMU_BOARD_ROLL))
        self.assertAlmostEqual(roll, 4.0, places=6)

    def test_no_offset_is_the_identity(self) -> None:
        v = (0.1, 0.9, -0.2)
        for component, expected in zip(unroll_imu_frame(v, board_roll=0.0), v):
            self.assertAlmostEqual(component, expected, places=9)


if __name__ == '__main__':
    unittest.main()
