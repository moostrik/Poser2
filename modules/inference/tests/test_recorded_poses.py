"""The recorded poses: the RAW stage's poses as the recorder keeps them, and `RecordedPoses` playing them back in
place of the pose model, after their latency, by track_id; without a GPU."""

import time
import unittest

import numpy as np

from modules.camera import FrameMetadata, FrameType
from modules.inference import RecordedPoses, RecordedPosesSettings, frame_poses
from modules.pose import BBox, Frame, FrameDict, FrameDictCallbackMixin, Points2D
from modules.utils import Rect

POINTS: np.ndarray = np.column_stack((np.linspace(0.1, 0.9, 17), np.full(17, 0.5), np.full(17, 0.8))).astype(np.float32)
BOX: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


def pose(track_id: int, timestamp: float, with_points: bool = False) -> Frame:
    features: dict = {BBox: BBox.from_rect(Rect(0.5, 0.5, 0.2, 0.2))}
    if with_points:
        features[Points2D] = Points2D(values=POINTS[:, :2].copy(), scores=POINTS[:, 2].copy())
    return Frame(track_id=track_id, cam_id=0, timestamp=timestamp, features=features)


class FakeInference(FrameDictCallbackMixin):
    """`PoseInference` without a GPU: the crop gives the poses back, the model is counted."""
    def __init__(self) -> None:
        super().__init__()
        self.model_runs: int = 0
        self.crops: int = 0

    def submit(self, frames: FrameDict) -> None:
        self.crops += 1
        self.model_runs += 1

    def cut_crops(self, frames: FrameDict) -> tuple[FrameDict, dict]:
        self.crops += 1
        return dict(frames), {}

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...


def wait_for(condition, seconds: float = 2.0) -> bool:
    end: float = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.002)
    return condition()


class FramePosesTest(unittest.TestCase):
    def test_the_poses_with_points_per_camera(self) -> None:
        frames: FrameDict = {0: pose(0, 2.0, with_points=True), 1: pose(1, 2.0)}       # track 1 has no points
        [kept] = frame_poses(frames, 2.05)
        self.assertEqual((kept.cam_id, kept.timestamp, kept.arrival, list(kept.poses)), (0, 2.0, 2.05, [0]))
        box, points = kept.poses[0]
        np.testing.assert_array_almost_equal(box, [0.5, 0.5, 0.2, 0.2])
        np.testing.assert_array_almost_equal(points, POINTS)


class RecordedPosesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RecordedPosesSettings()
        self.settings.initialize()
        self.inference = FakeInference()
        self.node = RecordedPoses(self.settings, self.inference)   # pyright: ignore[reportArgumentType]
        self.played: list[tuple[float, FrameDict]] = []
        self.node.add_frames_callback(lambda frames: self.played.append((time.perf_counter(), frames)))
        self.node.start()

    def tearDown(self) -> None:
        self.node.stop()

    def clip(self, captured: float, recorded: float, poses: dict | None, latency: float = 0.03) -> None:
        self.node.set_clip_frame(0, FrameType.VIDEO, np.zeros(1), FrameMetadata(recorded, poses=poses, pose_latency=latency), captured)

    def test_the_recorded_box_and_points_play_after_their_latency(self) -> None:
        captured = time.perf_counter()
        self.clip(captured, 10.0, {3: (BOX, POINTS)})
        self.node.submit({3: pose(3, captured)})
        self.assertTrue(wait_for(lambda: self.played))
        came, frames = self.played[0]
        self.assertGreaterEqual(came, captured + 0.029)
        rect = frames[3][BBox].to_rect()
        np.testing.assert_array_almost_equal([rect.x, rect.y, rect.width, rect.height], BOX)
        np.testing.assert_array_almost_equal(frames[3][Points2D].values, POINTS[:, :2])
        self.assertEqual((self.inference.model_runs, self.inference.crops), (0, 1))     # cropped, no model

    def test_a_track_id_the_recording_has_not_goes_on_without_points_and_is_counted(self) -> None:
        captured = time.perf_counter()
        self.clip(captured, 10.0, {3: (BOX, POINTS)}, 0.0)
        self.node.submit({3: pose(3, captured), 4: pose(4, captured)})
        self.assertTrue(wait_for(lambda: self.played))
        self.assertFalse(np.any(self.played[0][1][4][Points2D].scores > 0.0))
        self.assertEqual(self.node.missed.total('track_id'), 1)

    def test_a_frame_without_recorded_poses_sends_none(self) -> None:
        captured = time.perf_counter()
        self.clip(captured, 10.0, None, 0.0)
        self.node.submit({3: pose(3, captured)})
        time.sleep(0.05)
        self.assertEqual(self.played, [])

    def test_a_jump_drops_what_waits(self) -> None:
        captured = time.perf_counter()
        self.clip(captured, 10.0, {3: (BOX, POINTS)}, 0.2)
        self.node.submit({3: pose(3, captured)})
        self.node.clear_waiting()                                   # the player's jump: a loop
        self.clip(captured + 0.01, 2.0, {3: (BOX, POINTS)}, 0.0)
        self.node.submit({3: pose(3, captured + 0.01)})
        self.assertTrue(wait_for(lambda: self.played))
        time.sleep(0.25)
        self.assertEqual([frames[3].timestamp for _, frames in self.played], [captured + 0.01])

    def test_off_the_pose_model_runs_and_its_poses_go_on(self) -> None:
        self.settings.enabled = False
        self.node.submit({3: pose(3, 1.0)})
        self.assertEqual(self.inference.model_runs, 1)
        self.inference._notify_frames_callbacks({3: pose(3, 1.0, with_points=True)})
        self.assertEqual(len(self.played), 1)

    def test_on_the_pose_model_s_poses_do_not_go_on(self) -> None:
        self.inference._notify_frames_callbacks({3: pose(3, 1.0, with_points=True)})
        self.assertEqual(self.played, [])


if __name__ == '__main__':
    unittest.main()
