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
