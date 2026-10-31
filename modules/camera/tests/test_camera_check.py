"""What the camera check publishes for the pinned bar: the link the cameras really opened on."""

import unittest

from modules.camera import CameraCheck, CameraCheckSettings, RigCameraSettings


def published_link(*links: str) -> str:
    cameras: list[RigCameraSettings] = [RigCameraSettings() for _ in links]
    for camera, link in zip(cameras, links):
        camera.readings.link = link
    settings = CameraCheckSettings()
    CameraCheck(cameras, settings).update(0.0)
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


if __name__ == '__main__':
    unittest.main()
