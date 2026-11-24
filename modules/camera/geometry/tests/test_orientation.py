"""The camera readout: gravity to angles, and the mount readout that smooths them into the readings.

It needs no device: the gravity conversion is a pure function. How the gravity vector gets out of an OAK board's
IMU frame is the backend's (`modules/camera/oak/tests/test_imu.py`); the verdicts on the readings are the cameras'
health's (`modules/camera/tests/test_camera_health.py`).
"""

import math
import unittest

from modules.camera import RigCameraReadings
from modules.camera.geometry import MountReadout, orientation_from_gravity, unroll_imu_frame
from modules.camera.geometry.orientation import IMU_SMOOTHING


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


class TestOrientationFromGravity(unittest.TestCase):
    def test_level_camera(self) -> None:
        tilt, roll = orientation_from_gravity(0.0, 1.0, 0.0)
        self.assertAlmostEqual(tilt, 0.0, places=9)
        self.assertAlmostEqual(roll, 0.0, places=9)

    def test_aimed_at_the_sky(self) -> None:
        tilt, roll = orientation_from_gravity(0.0, 0.0, -1.0)
        self.assertAlmostEqual(tilt, 90.0, places=9)

    def test_pure_tilt_reports_no_roll(self) -> None:
        for angle in (-20.0, 0.0, 12.0, 16.0, 45.0):
            tilt, roll = orientation_from_gravity(*gravity(angle, 0.0))
            self.assertAlmostEqual(tilt, angle, places=6, msg=f'tilt {angle}')
            self.assertAlmostEqual(roll, 0.0, places=6, msg=f'tilt {angle}')

    def test_pure_roll_reports_no_tilt(self) -> None:
        for angle in (-30.0, 0.0, 20.0, 75.0):
            tilt, roll = orientation_from_gravity(*gravity(0.0, angle))
            self.assertAlmostEqual(tilt, 0.0, places=6, msg=f'roll {angle}')
            self.assertAlmostEqual(roll, angle, places=6, msg=f'roll {angle}')

    def test_the_two_axes_stay_separable_when_combined(self) -> None:
        """The property the whole readout exists for: a mount error is attributable to one axis
        or the other, instead of arriving as one number the panorama already cannot decompose."""
        for tilt_in in (-15.0, 0.0, 12.0, 16.0):
            for roll_in in (-10.0, 0.0, 5.0, 25.0):
                tilt, roll = orientation_from_gravity(*gravity(tilt_in, roll_in))
                self.assertAlmostEqual(tilt, tilt_in, places=6, msg=f'{tilt_in}/{roll_in}')
                self.assertAlmostEqual(roll, roll_in, places=6, msg=f'{tilt_in}/{roll_in}')

    def test_magnitude_does_not_matter(self) -> None:
        """The accelerometer reports m/s^2, not a unit vector, and is never scaled on the way in."""
        base = orientation_from_gravity(*gravity(16.0, 5.0))
        scaled = orientation_from_gravity(*(c * 9.81 for c in gravity(16.0, 5.0)))
        self.assertAlmostEqual(base[0], scaled[0], places=9)
        self.assertAlmostEqual(base[1], scaled[1], places=9)

    def test_a_dead_sensor_reads_nan_not_zero(self) -> None:
        tilt, roll = orientation_from_gravity(0.0, 0.0, 0.0)
        self.assertTrue(math.isnan(tilt) and math.isnan(roll))


class TestMountReadout(unittest.TestCase):
    def _readings(self, roll_offset: float = 0.0) -> RigCameraReadings:
        readings = RigCameraReadings()
        readings.initialize()
        readings.roll_offset = roll_offset
        return readings

    def test_no_sample_sets_nothing(self) -> None:
        readings = self._readings()
        self.assertIsNone(MountReadout(readings, 0.0).update([]))
        self.assertTrue(math.isnan(readings.tilt_measured))

    def test_the_first_sample_is_the_reading(self) -> None:
        readings = self._readings()
        tilt = MountReadout(readings, 0.0).update([gravity(10.0, 3.0)])
        assert tilt is not None
        self.assertAlmostEqual(tilt, 10.0, places=6)
        self.assertAlmostEqual(readings.tilt_measured, 10.0, places=6)
        self.assertAlmostEqual(readings.roll_measured, 3.0, places=6)

    def test_a_later_sample_pulls_the_average_by_the_smoothing(self) -> None:
        readout = MountReadout(self._readings(), 0.0)
        readout.update([(0.0, 1.0, 0.0)])
        readout.update([(1.0, 1.0, 0.0)])
        assert readout.gravity is not None
        self.assertAlmostEqual(readout.gravity[0], IMU_SMOOTHING, places=9)

    def test_the_roll_offset_and_the_board_roll_come_off_the_roll(self) -> None:
        readings = self._readings(roll_offset=2.0)
        MountReadout(readings, 90.0).update([unroll_imu_frame(gravity(0.0, 5.0), -90.0)])
        self.assertAlmostEqual(readings.roll_measured, 3.0, places=6)


if __name__ == '__main__':
    unittest.main()
