"""Depth in a video frame: the packing, and a round trip through the iGPU where there is one."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.video import StreamReader, StreamWriter, depth_frame_bytes, depth_to_p012, p012_to_depth
from modules.video.depth_frame import DEPTH_ENCODER, DEPTH_STEP_MM


class PackingTest(unittest.TestCase):
    def test_depth_comes_back_within_a_step_and_no_depth_stays_zero(self) -> None:
        depth = np.array([[0, 1500, 12000, 300], [4000, 0, 25, 9000]], np.uint16)
        raw = depth_to_p012(depth)
        self.assertEqual(len(raw), depth_frame_bytes(4, 2))
        back = p012_to_depth(raw, 4, 2)
        expected = np.where(depth < 100, 0, depth)
        self.assertTrue(np.all(np.abs(back.astype(int) - expected.astype(int)) < DEPTH_STEP_MM))

    def test_what_does_not_fit_stops_at_full_scale(self) -> None:
        depth = np.full((2, 2), 30000, np.uint16)
        self.assertEqual(int(p012_to_depth(depth_to_p012(depth), 2, 2).max()), 4095 * DEPTH_STEP_MM)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class RoundTripTest(unittest.TestCase):
    def test_depth_survives_the_igpu(self) -> None:
        frames: list[np.ndarray] = []
        for i in range(30):
            depth = np.full((300, 376), 4000, np.uint16)
            depth[100:200, 50 + i:150 + i] = 1500
            depth[250:280, 300:350] = 0
            frames.append(depth)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'depth.mkv'
            writer = StreamWriter(DEPTH_ENCODER, 1)
            writer.start(str(path), 60.0)
            time.sleep(0.05)
            for frame in frames:
                writer.submit_frame(frame)
            writer.stop()

            read: list[np.ndarray] = []
            reader = StreamReader(0, 'depth', lambda _c, _t, frame, _chunk, _id: read.append(frame.copy()), False, fps=1000.0, depth=True)
            reader.load(str(path), 0)
            while not reader.is_loaded():
                time.sleep(0.01)
            reader.play()
            while reader.is_playing():
                time.sleep(0.01)
            reader.join()

        self.assertEqual(len(read), len(frames))
        for frame, back in zip(frames, read):
            self.assertEqual(back.dtype, np.uint16)
            self.assertLess(np.percentile(np.abs(back.astype(int) - frame.astype(int)), 99), 40)
            self.assertEqual(int(back[255:275, 310:340].max()), 0)


if __name__ == '__main__':
    unittest.main()
