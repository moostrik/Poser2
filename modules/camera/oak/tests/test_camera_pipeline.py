"""Each OAK-D builds its own pipeline from its own settings at open, a clip camera too. Runs without a device: the
device and its list are stubbed, the pipeline and its warp mesh are depthai's own. On a depthai 2 install (`.venv`)."""

import unittest
from unittest.mock import MagicMock, patch

from modules.camera.oak import Camera, ClipCamera, OakCameraSettings
from modules.camera.oak import camera as oak_camera
from modules.camera.oak import pipeline as oak_pipeline

DEVICE_IDS: list[str] = ['A', 'B']


def settings(device_id: str, flip_h: bool) -> OakCameraSettings:
    return OakCameraSettings(device_id=device_id, flip_h=flip_h, color=False, square=False, yolo=False)


class CameraPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.meshes: list[list[tuple[float, float]]] = []
        real_find_warp = oak_pipeline.find_warp

        def find_warp(*args, **kwargs):
            mesh = real_find_warp(*args, **kwargs)
            self.meshes.append([(point.x, point.y) for point in mesh])
            return mesh
        for patcher in (patch.object(oak_camera, 'get_device_list', return_value=DEVICE_IDS),
                        patch.object(oak_camera.dai, 'Device', MagicMock()),
                        patch.object(oak_pipeline, 'find_warp', find_warp)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def open_all(self, cameras: list[Camera]) -> None:
        for camera in cameras:
            self.assertTrue(camera._open())
            self.addCleanup(camera._close)

    def test_two_cameras_build_two_pipelines_each_with_its_own_mesh(self) -> None:
        cameras: list[Camera] = [Camera(settings('A', False), 0), Camera(settings('B', True), 1)]
        self.open_all(cameras)
        self.assertIsNot(cameras[0].pipeline, cameras[1].pipeline)
        self.assertEqual(len(self.meshes), 2)
        self.assertNotEqual(self.meshes[0], self.meshes[1])

    def test_one_camera_builds_one_pipeline(self) -> None:
        camera = Camera(settings('A', False))
        self.open_all([camera])
        self.assertIsNotNone(camera.pipeline)
        self.assertEqual(len(self.meshes), 1)

    def test_clip_cameras_build_their_own_pipelines(self) -> None:
        player = MagicMock()
        cameras: list[Camera] = []
        for cam_id, flip_h in enumerate((False, True)):
            clip_settings = settings(DEVICE_IDS[cam_id], flip_h)
            clip_settings.clip_warp = True
            cameras.append(ClipCamera(player, clip_settings, cam_id))
        self.open_all(cameras)
        self.assertIsNot(cameras[0].pipeline, cameras[1].pipeline)
        self.assertEqual(len(self.meshes), 2)

    def test_close_drops_the_pipeline(self) -> None:
        camera = Camera(settings('A', False))
        self.assertTrue(camera._open())
        camera._close()
        self.assertIsNone(camera.pipeline)


if __name__ == '__main__':
    unittest.main()
