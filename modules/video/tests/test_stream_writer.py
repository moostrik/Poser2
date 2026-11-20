"""`StreamWriter`: a gray stream, as a camera's depth confidence is, through HEVC on the iGPU and back."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.video import StreamReader, StreamWriter

WIDTH, HEIGHT, FRAMES = 480, 300, 20


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class GrayStreamTest(unittest.TestCase):
    def test_a_gray_stream_comes_back_on_the_igpu(self) -> None:
        frames: list[np.ndarray] = []
        for i in range(FRAMES):
            frame = np.full((HEIGHT, WIDTH), 230, np.uint8)
            frame[100:200, 50 + i:150 + i] = 80
            frames.append(frame)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gray.mkv'
            writer = StreamWriter(video.ClipCodec.HEVC, video.Hardware.iGPU, keyframe_interval=1)
            writer.start(str(path), 50.0)
            time.sleep(0.05)
            for frame in frames:
                writer.submit_frame(frame)
            writer.stop()
            self.assertFalse(writer.failed)
            self.assertEqual(video.probe(str(path)).frames, FRAMES)

            read: list[np.ndarray] = []
            reader = StreamReader(0, 'gray', lambda _c, _t, frame, _chunk, _id: read.append(frame.copy()), False, fps=1000.0)
            reader.load(str(path), 0)
            while not reader.is_loaded():
                time.sleep(0.01)
            reader.play()
            while reader.is_playing():
                time.sleep(0.01)
            reader.join()

        self.assertEqual(len(read), FRAMES)
        for frame, back in zip(frames, read):
            self.assertEqual(back.shape, (HEIGHT, WIDTH))
            self.assertLess(float(np.abs(back.astype(np.int32) - frame).mean()), 3.0)


if __name__ == '__main__':
    unittest.main()
