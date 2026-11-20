"""NV12 and yuv444p frames through the writer as HEVC on NVENC, read back: count, size, codec, pixel format, color
tags and levels."""

import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from modules import video

SIZE, FRAMES, FPS = 256, 10, 30.0     # NVENC refuses frames of 128 wide and less


def yuv_gray(luma: int, rows: float) -> np.ndarray:
    """A flat gray frame of `rows` × SIZE rows (1.5 NV12, 3 yuv444p): Y at `luma`, chroma neutral."""
    frame = np.full((round(SIZE * rows), SIZE), 128, np.uint8)
    frame[:SIZE] = luma
    return frame


def nv12_gray(luma: int) -> np.ndarray:
    return yuv_gray(luma, 1.5)


def luma_of(i: int) -> int:
    return 40 + i * 16


def stream_info(path: str) -> dict:
    args = ['-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,codec_tag_string,color_space,color_range,pix_fmt', '-of', 'json', path]
    process = video.popen(args, subprocess.DEVNULL, subprocess.PIPE, command=video.FFPROBE)
    out, _ = process.communicate(timeout=20)
    return json.loads(out)['streams'][0]


def hevc_nvenc_works() -> bool:
    """A one-frame hevc_nvenc trial at the test's size."""
    args = ['-nostdin', '-f', 'lavfi', '-i', f'color=c=gray:size={SIZE}x{SIZE}:rate=15', '-frames:v', '1',
            '-c:v', 'hevc_nvenc', '-f', 'null', '-']
    try:
        process = video.popen(args, subprocess.DEVNULL, subprocess.DEVNULL)
        process.communicate(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0


@unittest.skipUnless(video.available() and hevc_nvenc_works(), "no ffmpeg with hevc_nvenc")
class TestClipHevc(unittest.TestCase):

    def test_nv12_frames_come_back_as_hevc(self) -> None:
        self._round_trip(video.ClipInput.NV12, 1.5, 'yuv420p')

    def test_yuv444p_frames_come_back_as_hevc_444(self) -> None:
        self._round_trip(video.ClipInput.YUV444P, 3.0, 'yuv444p')

    def _round_trip(self, clip_input: video.ClipInput, rows: float, pix_fmt: str) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clip.mp4')
            writer = video.ClipWriter(path, FPS, video.Encoder.NVENC, clip_input, video.ClipCodec.HEVC)
            for i in range(FRAMES):
                writer.add(yuv_gray(luma_of(i), rows))
            self.assertTrue(writer.close())

            frames, fps, width, height = video.probe(path)
            self.assertEqual((frames, width, height), (FRAMES, SIZE, SIZE))
            self.assertAlmostEqual(fps, FPS)
            info = stream_info(path)
            self.assertEqual((info['codec_name'], info['codec_tag_string']), ('hevc', 'hvc1'))
            self.assertEqual((info['color_space'], info['color_range']), ('bt709', 'tv'))
            self.assertEqual(info['pix_fmt'], pix_fmt)

            decoded, _ = video.decode_clip(path, (SIZE, SIZE))
            self.assertEqual(decoded.shape[0], FRAMES)
            for i in (0, FRAMES - 1):
                expected = (luma_of(i) - 16) * 255.0 / 219.0     # video-range gray to full range
                np.testing.assert_allclose(decoded[i].mean(axis=(0, 1)), [expected] * 3, atol=4)

    def test_the_nvenc_trial_passes(self) -> None:
        self.assertTrue(video.encoder_works(video.Encoder.NVENC))

    def test_nv12_shape_must_be_even_and_three_halves(self) -> None:
        writer = video.ClipWriter('unused.mp4', FPS, video.Encoder.NVENC, video.ClipInput.NV12, video.ClipCodec.HEVC)
        with self.assertRaises(ValueError):
            writer.add(np.zeros((SIZE, SIZE), np.uint8))


class TestClipCodecChoice(unittest.TestCase):

    def test_hevc_needs_nvenc(self) -> None:
        with self.assertRaises(ValueError):
            video.ClipWriter('unused.mp4', FPS, video.Encoder.CPU, codec=video.ClipCodec.HEVC)


if __name__ == '__main__':
    unittest.main()
