"""The mount readout: gravity to angles, and the verdict that summarises it.

Neither needs a device. The gravity conversion is a pure function, and `MountCheck` reads plain
settings objects, so the interesting cases — a rig with no IMU, one camera out of four off — are
all reachable without hardware.
"""

import math
import unittest

from modules.oak import MountCheck, MountCheckSettings, imu_to_camera, unroll_imu_frame, \
    orientation_from_gravity, mount_deviation
from modules.oak.camera.settings import CameraReadings, CameraSettings


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
                tilt, roll = orientation_from_gravity(*unroll_imu_frame(as_imu(gravity(tilt_in, roll_in))))
                self.assertAlmostEqual(tilt, tilt_in, places=6, msg=f'{tilt_in}/{roll_in}')
                self.assertAlmostEqual(roll, roll_in, places=6, msg=f'{tilt_in}/{roll_in}')

    def test_a_real_tripod_roll_is_not_swallowed(self) -> None:
        """The point of the correction: what is left after the quarter turn is the actual roll."""
        _, roll = orientation_from_gravity(*unroll_imu_frame(as_imu(gravity(12.0, 4.0))))
        self.assertAlmostEqual(roll, 4.0, places=6)

    def test_no_offset_is_the_identity(self) -> None:
        v = (0.1, 0.9, -0.2)
        for component, expected in zip(unroll_imu_frame(v, board_roll=0.0), v):
            self.assertAlmostEqual(component, expected, places=9)


def _cameras(readings: list[tuple[float, float]], configured_tilt: float = 12.0) -> list[CameraSettings]:
    cameras: list[CameraSettings] = []
    for tilt_measured, roll_measured in readings:
        camera = CameraSettings()
        camera.tilt = configured_tilt
        camera.readings.tilt_measured = tilt_measured
        camera.readings.roll_measured = roll_measured
        cameras.append(camera)
    return cameras


def _mount(readings: list[tuple[float, float]], tolerance: float = 2.0) -> bool:
    settings = MountCheckSettings()
    settings.tolerance = tolerance
    MountCheck(_cameras(readings), settings).update()
    return settings.mount


class TestMountCheck(unittest.TestCase):
    NAN: float = float('nan')

    def test_deviation_is_signed(self) -> None:
        tilt, roll = mount_deviation(_cameras([(10.5, -1.5)], configured_tilt=12.0)[0])
        self.assertAlmostEqual(tilt, -1.5)
        self.assertAlmostEqual(roll, -1.5)

    def test_deviation_is_nan_without_a_reading(self) -> None:
        tilt, roll = mount_deviation(_cameras([(self.NAN, self.NAN)])[0])
        self.assertTrue(math.isnan(tilt))
        self.assertTrue(math.isnan(roll))

    def test_no_readings_is_a_warning(self) -> None:
        self.assertFalse(_mount([(self.NAN, self.NAN)] * 4))

    def test_all_within_tolerance_is_ok(self) -> None:
        self.assertTrue(_mount([(12.5, 0.4), (11.0, -1.9), (12.0, 0.0), (13.9, 1.0)]))

    def test_one_tilt_past_tolerance_is_a_warning(self) -> None:
        self.assertFalse(_mount([(12.0, 0.0), (15.0, 0.0), (12.0, 0.0), (12.0, 0.0)]))

    def test_one_negative_roll_past_tolerance_is_a_warning(self) -> None:
        self.assertFalse(_mount([(12.0, 0.0), (12.0, -2.5), (12.0, 0.0), (12.0, 0.0)]))

    def test_a_camera_without_imu_does_not_spoil_ok(self) -> None:
        self.assertTrue(_mount([(12.0, 0.0), (self.NAN, self.NAN), (12.3, 0.5), (12.0, -0.2)]))


class TestRollOffsetIsABox(unittest.TestCase):
    """A box, not a slider. `Widget.resolve` sends a bounded float to `slider`, so this is one
    stray `min=` away from silently becoming one — worth a test rather than a comment."""

    def test_it_resolves_to_a_number_box(self) -> None:
        from modules.settings import Widget
        field = CameraReadings.roll_offset                       # type: ignore[attr-defined]
        self.assertEqual(Widget.resolve(field), Widget.number)
        self.assertIsNone(field.min)
        self.assertIsNone(field.max)

    def test_it_is_recorded_in_the_preset(self) -> None:
        """It is a measurement, so unlike the rest of `readings` it has to survive a save."""
        camera = CameraSettings()
        camera.readings.roll_offset = -2.2458989103710865
        self.assertIn('roll_offset', camera.to_dict()['readings'])
        self.assertEqual(camera.to_dict()['readings']['roll_offset'], -2.2458989103710865)

    def test_the_rest_of_readings_stays_out_of_the_preset(self) -> None:
        stored = CameraSettings().to_dict()['readings']
        for read_only in ('video_fps', 'tilt_measured', 'roll_measured', 'fov_factory'):
            self.assertNotIn(read_only, stored)


if __name__ == '__main__':
    unittest.main()
