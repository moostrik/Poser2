"""`FrameHandOver`: a camera's frame streams handed over together, in order, each camera with its own
streams, two chunks beside each other."""

import unittest

import numpy as np

from modules.camera import FrameMetadata, FrameType, PlayClock
from modules.camera.recording.player.frame_hand_over import FrameHandOver


def frames(times: list[float], latency: float = 0.03) -> list[FrameMetadata]:
    return [FrameMetadata(t, video_latency=latency) for t in times]


class FakeReaders:
    """A chunk's readers as the hand-over sees them: its number, each camera's streams, metadata and clock."""
    def __init__(self, chunk: int = 0, times: dict[int, list[float]] | None = None) -> None:
        self.chunk = chunk
        self.types = {0: {FrameType.VIDEO, FrameType.DEPTH}, 1: {FrameType.VIDEO}}
        self.frames = {c: frames(t) for c, t in (times or {}).items()}
        self.clock = PlayClock(self.frames, {c: 0 for c in self.frames})

    def frame(self, cam_id: int, frame_id: int) -> FrameMetadata | None:
        cam = self.frames.get(cam_id)
        return cam[frame_id] if cam is not None and 0 <= frame_id < len(cam) else None

    def info(self, cam_id: int) -> None:
        return None


class FrameHandOverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hand_over = FrameHandOver(2, [FrameType.VIDEO, FrameType.DEPTH])
        self.hand_over.start(FakeReaders(0, {0: [10.0, 10.02, 10.04, 10.06, 10.08, 10.10], 1: [10.0, 10.05, 10.1]}))  # pyright: ignore[reportArgumentType]
        self.out: list[tuple[int, FrameType]] = []
        self.out_cams: list[tuple[int, int, int]] = []
        self.hand_over.frame_callbacks.append(self.on_frame)

    def on_frame(self, cam: int, kind: FrameType, image: np.ndarray, metadata: FrameMetadata | None, captured: float | None) -> None:
        if cam == 0:
            self.out.append((int(image[0]), kind))
        self.out_cams.append((cam, int(image[1]), int(image[0])))

    def push(self, frame_id: int, kind: FrameType, chunk: int = 0, cam: int = 0) -> None:
        self.hand_over.submit_frame(cam, kind, np.array([frame_id, chunk]), chunk, frame_id)

    def test_a_frame_goes_out_with_both_its_streams(self) -> None:
        self.push(4, FrameType.VIDEO)
        self.assertEqual(self.out, [])
        self.push(4, FrameType.DEPTH)
        self.assertEqual(self.out, [(4, FrameType.VIDEO), (4, FrameType.DEPTH)])

    def test_a_frame_older_than_one_gone_out_is_dropped_and_counted(self) -> None:
        self.push(4, FrameType.VIDEO)
        self.push(5, FrameType.VIDEO)
        self.push(5, FrameType.DEPTH)       # 4 never gets its depth
        self.push(4, FrameType.DEPTH)       # too late: 5 went out
        self.assertEqual(self.out, [(5, FrameType.VIDEO), (5, FrameType.DEPTH)])
        self.assertEqual(self.hand_over.take_unpaired(), 1)
        self.assertEqual(self.hand_over.take_unpaired(), 0)

    def test_another_chunk_s_frames_are_not_handed_over(self) -> None:
        self.push(4, FrameType.VIDEO, chunk=1)
        self.push(4, FrameType.DEPTH, chunk=1)
        self.assertEqual(self.out, [])

    def test_a_camera_without_depth_goes_out_with_its_color(self) -> None:
        self.push(1, FrameType.VIDEO, cam=1)
        self.assertEqual(self.out_cams, [(1, 0, 1)])

    def test_the_next_chunk_plays_beside_the_one_ending_each_camera_in_order(self) -> None:
        self.hand_over.follow(FakeReaders(1, {0: [10.12], 1: [10.15]}))  # pyright: ignore[reportArgumentType]
        self.push(0, FrameType.VIDEO, chunk=1, cam=1)       # camera 1's chunk ended first: its next goes out
        self.push(5, FrameType.VIDEO, chunk=0)              # camera 0 still plays the first
        self.push(5, FrameType.DEPTH, chunk=0)
        self.push(2, FrameType.VIDEO, chunk=0, cam=1)       # older than camera 1's frame gone out: dropped
        self.assertEqual(self.out_cams, [(1, 1, 0), (0, 0, 5), (0, 0, 5)])

    def test_a_retired_chunk_s_frames_are_not_handed_over(self) -> None:
        readers = FakeReaders(1, {0: [10.12]})
        self.hand_over.follow(readers)  # pyright: ignore[reportArgumentType]
        self.hand_over.retire(readers)  # pyright: ignore[reportArgumentType]
        self.push(0, FrameType.VIDEO, chunk=1, cam=1)
        self.assertEqual(self.out_cams, [])

    def test_drift_is_the_recorded_time_between_the_cameras_in_ms(self) -> None:
        self.push(2, FrameType.VIDEO)
        self.push(2, FrameType.DEPTH)       # camera 0 at 10.04
        self.push(1, FrameType.VIDEO, cam=1)    # camera 1 at 10.05
        self.assertAlmostEqual(self.hand_over.drift, 10.0)

    def test_stopped_nothing_is_handed_over(self) -> None:
        self.hand_over.start(None)
        self.push(4, FrameType.VIDEO)
        self.push(4, FrameType.DEPTH)
        self.assertEqual(self.out, [])


if __name__ == '__main__':
    unittest.main()
