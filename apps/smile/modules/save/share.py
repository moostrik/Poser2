"""The share clip: the visitor's grin as the phone gets it to keep and share. The players' clip (silent,
`grins/<style>/<id>.mp4`) with the visitor's track as its sound, tailored to the clip's length by the
tape (modules/audio/tape.py: spun up over the first half second, wound down over the last) and a small
watermark in the bottom right, a circled "ai" (the image is the machine's) and the artist's name; no
caption, the joke is the room's. Built by the saver before the players' clip lands, the phone comes first."""

import logging

import cv2
import freetype
import numpy as np

from modules.audio import tape
from modules.credit_scroll import FONT, _render
from modules.video import ffmpeg
from modules.video.ffmpeg import Encoder

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 48000
WATERMARK_PX: int = 28          # the watermark's ink height, on the 896 x 1120 clip
WATERMARK_ALPHA: float = 0.8
SOURCE_MARGIN: float = 1.4      # the track read is this much longer than the clip: the spin-up runs at up to 1.5x


def watermark_image(text: str, height_px: int = WATERMARK_PX, alpha: float = WATERMARK_ALPHA, scale: int = 4) -> np.ndarray:
    """The watermark as a BGRA image: a ring with "ai" in it (the image is the machine's) and the artist's
    name after it, in the credit's font, white with one black edge round all of it (drawn at `scale` times
    the size, the edge grown like the screen's text, then scaled down)."""
    face = freetype.Face(FONT)
    size: int = height_px * scale
    stroke: int = max(2, round(size * 0.055))
    gap: int = round(size * 0.35)
    name: np.ndarray = _render(face, text, round(size * 0.9)) if text.strip() else np.zeros((0, 0), np.uint8)
    width: int = size + (gap + name.shape[1] if name.size else 0)
    ink: np.ndarray = np.zeros((size, width), np.uint8)
    centre: tuple[int, int] = (size // 2, size // 2)
    cv2.circle(ink, centre, size // 2 - stroke, 255, stroke, cv2.LINE_AA)
    letters: np.ndarray = _render(face, 'ai', round(size * 0.62))
    h, w = letters.shape[:2]
    y0: int = centre[1] - h // 2 + round(size * 0.01)
    x0: int = centre[0] - w // 2
    if 0 <= y0 and y0 + h <= size and 0 <= x0 and x0 + w <= size:
        ink[y0:y0 + h, x0:x0 + w] = np.maximum(ink[y0:y0 + h, x0:x0 + w], letters)
    if name.size:
        h, w = name.shape[:2]
        y0 = max(0, (size - h) // 2)
        ink[y0:y0 + min(h, size - y0), size + gap:size + gap + w] = name[:size - y0]
    outline: int = max(1, round(height_px * 0.025 * scale))
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
    padded: np.ndarray = np.pad(ink, outline)
    edge: np.ndarray = cv2.dilate(padded, kernel)
    mark: np.ndarray = np.zeros((padded.shape[0], padded.shape[1], 4), np.uint8)
    mark[:, :, 3] = edge
    for channel in range(3):
        mark[:, :, channel] = padded
    small: tuple[int, int] = (round(mark.shape[1] / scale), round(mark.shape[0] / scale))
    image: np.ndarray = cv2.resize(mark, small, interpolation=cv2.INTER_AREA)
    image[:, :, 3] = (image[:, :, 3].astype(np.float32) * alpha).astype(np.uint8)
    return image


def write_watermark(text: str, path: str) -> bool:
    """The watermark as a PNG file for ffmpeg; False when it could not be made."""
    try:
        return bool(cv2.imwrite(path, watermark_image(text)))
    except Exception:
        logger.exception('The watermark could not be made')
        return False


def build_share(clip: str, track: str | None, offset: float, watermark: str | None, encoder: Encoder, out: str,
                sample_rate: int = SAMPLE_RATE) -> None:
    """Write the share clip `out` from the players' clip: `track` from `offset` s (the point the room heard
    when the grin started) on the tape, or silence without one; the `watermark` PNG (or None) in the bottom
    right. Raises when it fails."""
    frames, fps, width, height = ffmpeg.probe(clip)
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f'{clip} does not say its length')
    duration: float = frames / fps
    seconds: float = duration * SOURCE_MARGIN + 1.0
    if track:
        audio: np.ndarray = ffmpeg.read_audio(track, offset, seconds, sample_rate)
    else:
        audio = np.zeros((round(seconds * sample_rate), 2), np.float32)
    pcm: np.ndarray = tape.resample(audio, tape.speed_curve(duration, sample_rate), sample_rate)
    overlays: list[tuple[str, str]] = [(watermark, 'W-w-24:H-h-24')] if watermark else []
    ffmpeg.make_share(clip, overlays, pcm, sample_rate, encoder, out)
