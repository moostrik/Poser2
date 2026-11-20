"""The YUV inputs (NV12, yuv444p and their 10-bit P010, xv30le) through the writer as HEVC on NVENC, read back:
count, size, codec, pixel format, color tags and levels; a 10-bit gradient comes back finer than 8-bit steps."""

import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from modules import video

SIZE, FRAMES, FPS = 256, 10, 30.0     # NVENC refuses frames of 128 wide and less


def yuv_gray(luma: int, rows: float, dtype: type[np.unsignedinteger] = np.uint8) -> np.ndarray:
    """A flat gray frame of `rows` × SIZE rows (1.5 for 4:2:0, 3 for 4:4:4): Y at the 8-bit code `luma`, chroma
    neutral; in uint16 the codes × 256, as 10 bits in the high bits."""
    scale = 1 if dtype is np.uint8 else 256
    frame = np.full((round(SIZE * rows), SIZE), 128 * scale, dtype)
    frame[:SIZE] = luma * scale
    return frame


def xv30(luma: np.ndarray | int, cb: int = 512, cr: int = 512) -> np.ndarray | int:
    """10-bit codes packed as xv30le: Cb in bits 0-9, Y in 10-19, Cr in 20-29."""
    return cb | (np.asarray(luma, np.uint32) << 10) | (cr << 20)


def xv30_gray(luma: int) -> np.ndarray:
    """A flat gray xv30le frame, SIZE × SIZE: Y at the 8-bit code `luma`, chroma neutral."""
    return np.full((SIZE, SIZE), xv30(luma * 4), np.uint32)


GRADIENT_WIDTH = 1024
RAMP = np.round(64 + np.arange(GRADIENT_WIDTH) * 876 / (GRADIENT_WIDTH - 1)).astype(np.uint32)


def yuv_gradient(clip_input: video.ClipInput) -> np.ndarray:
    """A 10-bit frame, GRADIENT_WIDTH × SIZE: Y from code 64 to 940 left to right, one 10-bit step a column or less,
    chroma neutral; P010 in uint16, or xv30le in uint32."""
    if clip_input is video.ClipInput.XV30:
        return np.repeat(np.asarray(xv30(RAMP), np.uint32)[None, :], SIZE, axis=0)
    frame = np.full((round(SIZE * 1.5), GRADIENT_WIDTH), 512 << 6, np.uint16)
    frame[:SIZE] = (RAMP << 6).astype(np.uint16)
    return frame


def decoded_luma_10bit(path: str) -> np.ndarray:
    """The first frame's Y plane as 10-bit video-range codes (yuv420p10le keeps the range, gray10le would not)."""
    args = ['-v', 'error', '-i', path, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', 'pipe:1']
    process = video.popen(args, subprocess.DEVNULL, subprocess.PIPE)
    out, _ = process.communicate(timeout=20)
    return np.frombuffer(out, np.uint16)[:SIZE * GRADIENT_WIDTH].reshape(SIZE, GRADIENT_WIDTH)


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

    def test_p010_frames_come_back_as_hevc_main_10(self) -> None:
        self._round_trip(video.ClipInput.P010, 1.5, 'yuv420p10le', np.uint16)

    def test_xv30_frames_come_back_as_hevc_444_10_bit(self) -> None:
        self._round_trip(video.ClipInput.XV30, 1.0, 'yuv444p10le', np.uint32)

    def test_a_10_bit_gradient_comes_back_finer_than_8_bit_steps(self) -> None:
        for clip_input in (video.ClipInput.P010, video.ClipInput.XV30):
            with self.subTest(clip_input=clip_input.name), tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, 'gradient.mp4')
                writer = video.ClipWriter(path, FPS, video.Hardware.GPU, clip_input, video.ClipCodec.HEVC)
                for _ in range(FRAMES):
                    writer.add(yuv_gradient(clip_input))
                self.assertTrue(writer.close())
                row = decoded_luma_10bit(path)[SIZE // 2].astype(np.int32)
                self.assertGreater(len(np.unique(row)), 600)            # 8 bits hold 220 Y codes
                self.assertGreater(float(np.mean(row % 4 != 0)), 0.5)   # 8-bit codes come back as multiples of 4
                self.assertLess(float(np.abs(row - RAMP.astype(np.int32)).max()), 4.0)

    def _round_trip(self, clip_input: video.ClipInput, rows: float, pix_fmt: str,
                    dtype: type[np.unsignedinteger] = np.uint8) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clip.mp4')
            writer = video.ClipWriter(path, FPS, video.Hardware.GPU, clip_input, video.ClipCodec.HEVC)
            for i in range(FRAMES):
                writer.add(xv30_gray(luma_of(i)) if clip_input is video.ClipInput.XV30
                           else yuv_gray(luma_of(i), rows, dtype))
            self.assertTrue(writer.close())

            probed = video.probe(path)
            self.assertEqual((probed.frames, probed.width, probed.height), (FRAMES, SIZE, SIZE))
            self.assertAlmostEqual(probed.fps, FPS)
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
        self.assertTrue(video.encoder_works(video.Hardware.GPU))

    def test_nv12_shape_must_be_even_and_three_halves(self) -> None:
        writer = video.ClipWriter('unused.mp4', FPS, video.Hardware.GPU, video.ClipInput.NV12, video.ClipCodec.HEVC)
        with self.assertRaises(ValueError):
            writer.add(np.zeros((SIZE, SIZE), np.uint8))


class TestClipCodecChoice(unittest.TestCase):

    def test_hevc_needs_nvenc(self) -> None:
        with self.assertRaises(ValueError):
            video.ClipWriter('unused.mp4', FPS, video.Hardware.CPU, codec=video.ClipCodec.HEVC)


if __name__ == '__main__':
    unittest.main()
