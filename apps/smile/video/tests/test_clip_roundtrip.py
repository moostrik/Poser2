"""Round trip of a clip through the writer and the decoder, on every encoder this machine has."""

import os
import tempfile
import unittest

import numpy as np

from apps.smile.video import ffmpeg
from apps.smile.video.ffmpeg import ClipWriter, Encoder

WIDTH, HEIGHT, FRAMES, FPS = 896, 1120, 20, 13.0
CELL = (512, 640)


def synthetic_frame(i: int) -> np.ndarray:
    """A flat colour that shifts per frame, with a gradient: easy to check, not trivial to encode."""
    frame = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    frame[:, :, 0] = 40 + i * 8                              # blue rises with the frame index
    frame[:, :, 1] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    frame[:, :, 2] = 200 - i * 4
    return frame


@unittest.skipUnless(ffmpeg.available(), 'ffmpeg not on the PATH')
class TestClipRoundTrip(unittest.TestCase):

    def test_roundtrip_on_every_working_encoder(self):
        encoders = [e for e in (Encoder.QSV, Encoder.NVENC, Encoder.CPU) if ffmpeg.encoder_works(e)]
        self.assertIn(Encoder.CPU, encoders)
        for encoder in encoders:
            with self.subTest(encoder=encoder.value):
                self._roundtrip(encoder)

    def _roundtrip(self, encoder: Encoder) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clip.mp4')
            writer = ClipWriter(path, FPS, encoder)
            for i in range(FRAMES):
                writer.add(synthetic_frame(i))
            self.assertTrue(writer.close())
            self.assertEqual(writer.frames, FRAMES)

            frames, fps = ffmpeg.decode_clip(path, CELL)
            self.assertEqual(frames.shape, (FRAMES, CELL[1], CELL[0], 3))
            self.assertAlmostEqual(fps, FPS)
            for i in (0, FRAMES - 1):
                expected = synthetic_frame(i).reshape(-1, 3).mean(axis=0)
                got = frames[i].reshape(-1, 3).mean(axis=0)
                np.testing.assert_allclose(got, expected, atol=6, err_msg=f'{encoder.value} frame {i}')

    def test_odd_size_is_cropped_to_even(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'odd.mp4')
            writer = ClipWriter(path, 15.0, Encoder.CPU)
            writer.add(np.full((33, 65, 3), 128, np.uint8))
            self.assertTrue(writer.close())
            _, _, width, height = ffmpeg.probe(path)
            self.assertEqual((width, height), (64, 32))

    def test_size_change_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = ClipWriter(os.path.join(folder, 'size.mp4'), 15.0, Encoder.CPU)
            writer.add(np.zeros((64, 64, 3), np.uint8))
            with self.assertRaises(ValueError):
                writer.add(np.zeros((32, 64, 3), np.uint8))
            writer.close()

    def test_pick_encoder_falls_back_to_cpu(self):
        self.assertIn(ffmpeg.pick_encoder(Encoder.AUTO), (Encoder.QSV, Encoder.NVENC, Encoder.CPU))
        self.assertEqual(ffmpeg.pick_encoder(Encoder.CPU), Encoder.CPU)
