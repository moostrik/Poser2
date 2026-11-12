"""`StreamReader`: a range starts at its frame, and no decoder outlives a reader."""

import os
import tempfile
import time
import unittest

import numpy as np

from modules import video
from modules.video import StreamReader


def make_clip(path: str, frames: int) -> None:
    writer = video.ClipWriter(path, 30.0, video.Encoder.CPU)
    for _ in range(frames):
        writer.add(np.zeros((48, 64, 3), np.uint8))
    writer.close()


def wait_for(condition, seconds: float = 5.0) -> bool:
    deadline: float = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


class FixedSchedule:
    """Frame `n` due at `origin + offsets[n]`; `delay` moves the origin and sums what it moved."""
    def __init__(self, origin: float, offsets: list[float]) -> None:
        self.origin: float = origin
        self.offsets: list[float] = offsets
        self.delayed: float = 0.0

    def moment(self, cam_id: int, frame_id: int) -> float | None:
        return self.origin + self.offsets[frame_id] if frame_id < len(self.offsets) else None

    def catch_up(self, cam_id: int, frame_id: int, now: float) -> float:
        due: float = self.origin + self.offsets[frame_id]
        late: float = max(now - due, 0.0)
        self.origin += late
        self.delayed += late
        return due + late


@unittest.skipUnless(video.available(), 'ffmpeg not on the PATH')
class StreamReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path: str = os.path.join(self.folder.name, 'clip.mp4')
        make_clip(self.path, 30)
        self.frames: list[int] = []

    def tearDown(self) -> None:
        self.folder.cleanup()

    def reader(self, fps: float = 1000.0) -> StreamReader:
        return StreamReader(0, 'video', lambda _c, _t, _f, _chunk, frame_id: self.frames.append(frame_id), True, fps=fps)

    def test_a_reader_stopped_before_it_plays_ends_its_decoder(self) -> None:
        reader = self.reader()
        reader.load(self.path, 0)
        self.assertTrue(wait_for(reader.is_loaded))
        process = reader.ffmpeg_process
        reader.stop()
        reader.join()
        self.assertIsNotNone(process.poll())
        self.assertFalse(reader.is_loaded())

    def test_a_reader_stopped_during_its_load_ends_its_decoder(self) -> None:
        reader = self.reader()
        reader.load(self.path, 0)
        reader.stop()
        reader.join()
        self.assertFalse(reader.is_loaded())

    def test_a_schedule_sets_when_each_frame_comes_never_out_of_order(self) -> None:
        schedule = FixedSchedule(time.perf_counter() + 0.5, [0.0, 0.02, 0.01, 0.04])     # frame 2 due before frame 1
        came: list[tuple[int, float]] = []
        reader = StreamReader(0, 'video', lambda _c, _t, _f, _chunk, frame_id: came.append((frame_id, time.perf_counter())), True,
                              fps=30.0, schedule=schedule)
        reader.load(self.path, 0, 0, 4)
        self.assertTrue(wait_for(reader.is_loaded))
        reader.play()
        self.assertTrue(wait_for(lambda: not reader.is_playing()))
        reader.join()
        self.assertEqual([frame for frame, _ in came], [0, 1, 2, 3])
        self.assertGreaterEqual(came[1][1], schedule.origin + 0.02 - 0.002)
        self.assertGreaterEqual(came[2][1], came[1][1])
        self.assertGreaterEqual(came[3][1], schedule.origin + 0.04 - 0.002)

    def test_a_reader_behind_its_schedule_delays_it(self) -> None:
        schedule = FixedSchedule(time.perf_counter() - 1.0, [0.0, 0.01])      # both a second overdue
        reader = StreamReader(0, 'video', lambda _c, _t, _f, _chunk, frame_id: None, True, fps=1000.0, schedule=schedule)
        reader.load(self.path, 0, 0, 2)
        self.assertTrue(wait_for(reader.is_loaded))
        reader.play()
        self.assertTrue(wait_for(lambda: not reader.is_playing()))
        reader.join()
        self.assertGreater(schedule.delayed, 0.9)

    def test_a_stop_ends_a_reader_waiting_on_its_decoder(self) -> None:
        schedule = FixedSchedule(time.perf_counter() + 60.0, [0.0] * 30)     # its first frame a minute away
        reader = StreamReader(0, 'video', lambda *_: None, True, fps=30.0, schedule=schedule)
        reader.load(self.path, 0)
        self.assertTrue(wait_for(reader.is_loaded))
        reader.play()
        time.sleep(0.1)
        process = reader.ffmpeg_process
        reader.stop()
        self.assertTrue(wait_for(lambda: process is not None and process.poll() is not None))
        self.assertTrue(wait_for(lambda: not reader.is_playing()))       # the reader's thread ends too

    def test_nv12_comes_packed_at_half_the_bytes(self) -> None:
        shapes: list[tuple[int, ...]] = []
        reader = StreamReader(0, 'video', lambda _c, _t, frame, _chunk, _i: shapes.append(frame.shape), True, fps=1000.0, nv12=True)
        reader.load(self.path, 0, 0, 2)
        self.assertTrue(wait_for(reader.is_loaded))
        reader.play()
        self.assertTrue(wait_for(lambda: not reader.is_playing()))
        reader.join()
        self.assertEqual(shapes, [(72, 64), (72, 64)])

    def test_a_range_counts_from_its_first_frame(self) -> None:
        reader = self.reader()
        reader.load(self.path, 0, 12, 15)
        self.assertTrue(wait_for(reader.is_loaded))
        reader.play()
        self.assertTrue(wait_for(lambda: not reader.is_playing()))
        reader.join()
        self.assertEqual(self.frames, [12, 13, 14])


if __name__ == '__main__':
    unittest.main()
