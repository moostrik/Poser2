"""The camera keeps trying: a failed open waits and tries again, with its callbacks kept, and frames
that stop open the camera again. Runs without a device. On a depthai 3 install (`.venv-oak4`)."""

import time
import unittest
from unittest.mock import patch

from modules.camera.oak4 import Oak4CameraSettings
from modules.camera.oak4.camera import Camera

CAMERA: str = 'modules.camera.oak4.camera'


class FakeOpenCamera(Camera):
    """Opens without a device, with the time of its last frame set by the test."""

    def __init__(self, last_frame_offset: float, stop_after: int) -> None:
        super().__init__(Oak4CameraSettings())
        self.last_frame_offset: float = last_frame_offset
        self.stop_after: int = stop_after
        self.opens: int = 0
        self.closes: int = 0

    def _open(self) -> bool:
        self.opens += 1
        if self.opens >= self.stop_after:
            self.stop()
        self._last_frame = time.perf_counter() + self.last_frame_offset
        return True

    def _close(self) -> None:
        self.closes += 1


@patch(f'{CAMERA}.RETRY_INTERVAL', 0.001)
@patch(f'{CAMERA}.FRAME_CHECK', 0.001)
class RetryTest(unittest.TestCase):
    def test_a_failed_open_tries_again_with_its_callbacks_until_stopped(self) -> None:
        camera = Camera(Oak4CameraSettings())
        callback = lambda frame: None
        camera.add_frame_callback(callback)
        callbacks_at_try: list[int] = []

        def no_device(verbose: bool) -> list[str]:
            callbacks_at_try.append(len(camera.frame_callbacks))
            if len(callbacks_at_try) == 3:
                camera.stop()
            return []

        with patch(f'{CAMERA}.get_device_list', no_device):
            camera.run()
        self.assertEqual(callbacks_at_try, [1, 1, 1])
        self.assertEqual(camera.frame_callbacks, [])

    def test_a_problem_logs_once(self) -> None:
        camera = Camera(Oak4CameraSettings())
        tries: list[int] = []

        def no_device(verbose: bool) -> list[str]:
            tries.append(1)
            if len(tries) == 3:
                camera.stop()
            return []

        with patch(f'{CAMERA}.get_device_list', no_device), self.assertLogs(CAMERA, 'WARNING') as logs:
            camera.run()
        self.assertEqual(len(logs.records), 1)

    def test_a_search_that_raises_does_not_end_the_thread(self) -> None:
        camera = Camera(Oak4CameraSettings())
        tries: list[int] = []

        def broken(verbose: bool) -> list[str]:
            tries.append(1)
            if len(tries) == 2:
                camera.stop()
            raise RuntimeError('no search')

        with patch(f'{CAMERA}.get_device_list', broken), self.assertLogs(CAMERA, 'WARNING'):
            camera.run()
        self.assertEqual(len(tries), 2)

    def test_a_device_that_will_not_open_is_tried_again(self) -> None:
        camera = Camera(Oak4CameraSettings())
        tries: list[int] = []

        def busy(info: object) -> None:
            tries.append(1)
            if len(tries) == 2:
                camera.stop()
            raise RuntimeError('in use')

        with patch(f'{CAMERA}.get_device_list', lambda verbose: ['x']), patch(f'{CAMERA}.find_device'), \
             patch(f'{CAMERA}.dai.Device', busy), self.assertLogs(CAMERA, 'WARNING'):
            camera.run()
        self.assertEqual(len(tries), 2)

    def test_a_refused_pipeline_is_not_tried_again(self) -> None:
        camera = Camera(Oak4CameraSettings())
        camera.add_frame_callback(lambda frame: None)
        builds: list[int] = []

        def refused(*args: object, **kwargs: object) -> None:
            builds.append(1)
            raise RuntimeError('no model')

        with patch(f'{CAMERA}.get_device_list', lambda verbose: ['x']), patch(f'{CAMERA}.find_device'), \
             patch(f'{CAMERA}.dai.Device'), patch(f'{CAMERA}.dai.Pipeline'), patch(f'{CAMERA}.build_pipeline', refused), \
             self.assertLogs(CAMERA, 'ERROR'):
            camera.run()
        self.assertEqual(len(builds), 1)
        self.assertFalse(camera.connected)
        self.assertEqual(camera.frame_callbacks, [])

    @patch(f'{CAMERA}.FRAME_TIMEOUT', 0.01)
    def test_frames_that_stop_open_the_camera_again(self) -> None:
        camera = FakeOpenCamera(last_frame_offset=0.0, stop_after=2)
        with self.assertLogs(CAMERA, 'WARNING'):
            camera.run()
        self.assertEqual((camera.opens, camera.closes), (2, 2))

    def test_frames_that_come_keep_it_open(self) -> None:
        camera = FakeOpenCamera(last_frame_offset=100.0, stop_after=2)
        camera.start()
        time.sleep(0.05)
        opens: int = camera.opens
        camera.stop()
        camera.join(timeout=2)
        self.assertEqual(opens, 1)


class ConnectedTest(unittest.TestCase):
    def test_a_camera_that_never_opened_is_not_connected(self) -> None:
        self.assertFalse(Camera(Oak4CameraSettings()).connected)

    def test_a_closed_camera_is_not_running_and_keeps_its_callbacks(self) -> None:
        camera = Camera(Oak4CameraSettings())
        camera.running = True
        camera.frame_callbacks.append(lambda frame: None)
        camera._close()
        self.assertFalse(camera.running)
        self.assertEqual(len(camera.frame_callbacks), 1)


if __name__ == '__main__':
    unittest.main()
