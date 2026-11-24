"""What the cameras' health publishes for the pinned bar: the link the cameras really opened on, the mount and
rate verdicts, and the slowest latency. It reads plain settings objects, so the interesting cases (a rig with no IMU,
one camera out of four off) need no hardware."""

import math
import unittest

from modules.camera import CameraHealth, CameraHealthSettings, RigCameraSettings, mount_deviation, fps_deviation
from modules.settings import Field


def published_link(*links: str) -> str:
    cameras: list[RigCameraSettings] = [RigCameraSettings() for _ in links]
    for camera, link in zip(cameras, links):
        camera.readings.link = link
    settings = CameraHealthSettings()
    CameraHealth(cameras, settings).update(0.0)
    return settings.link


class PublishedLinkTest(unittest.TestCase):
    def test_one_camera_shows_its_link(self) -> None:
        self.assertEqual(published_link('USB'), 'USB')

    def test_cameras_on_one_link_show_it_once(self) -> None:
        self.assertEqual(published_link('USB', 'USB'), 'USB')

    def test_cameras_on_two_links_show_both(self) -> None:
        self.assertEqual(published_link('USB', 'ETHERNET'), 'ETHERNET, USB')

    def test_a_backend_without_a_link_shows_none(self) -> None:
        self.assertEqual(published_link(''), '')


class _TiltedCamera(RigCameraSettings):
    """A backend with a configured mount tilt, as the OAK-D's settings have."""
    tilt: Field[float] = Field(0.0)

    @property
    def expected_tilt(self) -> float:
        return self.tilt


def _cameras(readings: list[tuple[float, float]], configured_tilt: float = 12.0) -> list[RigCameraSettings]:
    cameras: list[RigCameraSettings] = []
    for tilt_measured, roll_measured in readings:
        camera = _TiltedCamera()
        camera.tilt = configured_tilt
        camera.readings.tilt_measured = tilt_measured
        camera.readings.roll_measured = roll_measured
        cameras.append(camera)
    return cameras


def _mount(readings: list[tuple[float, float]], tolerance: float = 2.0) -> bool:
    settings = CameraHealthSettings()
    settings.mount_tolerance = tolerance
    CameraHealth(_cameras(readings), settings).update(0.0)
    return settings.mount


def _camera_fps(rates: list[float], configured_fps: float = 30.0) -> bool:
    cameras: list[RigCameraSettings] = []
    for rate in rates:
        camera = RigCameraSettings()
        camera.fps = configured_fps
        camera.readings.video_fps = rate
        cameras.append(camera)
    settings = CameraHealthSettings()
    CameraHealth(cameras, settings).update(0.0)
    return settings.camera_fps


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

    def test_no_imu_anywhere_is_not_a_warning(self) -> None:
        self.assertTrue(_mount([(self.NAN, self.NAN)] * 4))

    def test_all_within_tolerance_is_ok(self) -> None:
        self.assertTrue(_mount([(12.5, 0.4), (11.0, -1.9), (12.0, 0.0), (13.9, 1.0)]))

    def test_one_tilt_past_tolerance_is_a_warning(self) -> None:
        self.assertFalse(_mount([(12.0, 0.0), (15.0, 0.0), (12.0, 0.0), (12.0, 0.0)]))

    def test_one_negative_roll_past_tolerance_is_a_warning(self) -> None:
        self.assertFalse(_mount([(12.0, 0.0), (12.0, -2.5), (12.0, 0.0), (12.0, 0.0)]))

    def test_a_camera_without_imu_does_not_spoil_ok(self) -> None:
        self.assertTrue(_mount([(12.0, 0.0), (self.NAN, self.NAN), (12.3, 0.5), (12.0, -0.2)]))


class TestCameraFpsCheck(unittest.TestCase):
    def test_deviation_is_a_signed_fraction(self) -> None:
        camera = RigCameraSettings()
        camera.fps = 30.0
        camera.readings.video_fps = 27.0
        self.assertAlmostEqual(fps_deviation(camera), -0.1)

    def test_within_five_percent_is_ok(self) -> None:
        self.assertTrue(_camera_fps([30.0, 28.6, 31.4]))

    def test_one_camera_slow_is_a_warning(self) -> None:
        self.assertFalse(_camera_fps([30.0, 28.4, 30.0]))

    def test_one_camera_fast_is_a_warning(self) -> None:
        self.assertFalse(_camera_fps([30.0, 31.6, 30.0]))

    def test_no_frames_is_a_warning(self) -> None:
        self.assertFalse(_camera_fps([0.0]))


class TestPinnedLatency(unittest.TestCase):
    def test_the_slowest_camera_is_published(self) -> None:
        cameras: list[RigCameraSettings] = []
        for video, depth in [(26.0, 47.0), (28.0, 44.0)]:
            camera = RigCameraSettings()
            camera.readings.video_latency = video
            camera.readings.depth_latency = depth
            cameras.append(camera)
        settings = CameraHealthSettings()
        CameraHealth(cameras, settings).update(0.0)
        self.assertEqual(settings.video_latency, 28.0)
        self.assertEqual(settings.depth_latency, 47.0)


if __name__ == '__main__':
    unittest.main()
