"""The share clip: the visitor's grin as the phone gets it to keep and share. The players' clip (silent,
`grins/<style>/<id>.mp4`) with the visitor's track as its sound, tailored to the clip's length by the
tape (modules/audio/tape.py: spun up over the first half second, wound down over the last) and a small
watermark in the bottom right, "SMILE", a circled "ai" (the machine's mark) and "by" the artist's name; no
caption, the joke is the room's. Built by the saver before the players' clip lands, the phone comes first."""

import logging

import cv2
import numpy as np

from ..audio import resample, speed_curve
from ..credit import title_image
from modules import video

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 48000
WATERMARK_PX: int = 28          # the watermark's ink height, on the 896 x 1120 clip
WATERMARK_ALPHA: float = 0.8
SOURCE_MARGIN: float = 1.4      # the track read is this much longer than the clip: the spin-up runs at up to 1.5x


def write_watermark(text: str, path: str) -> bool:
    """The watermark, the machine's title with "by" the artist's name, as a PNG file for ffmpeg; False when it
    could not be made."""
    try:
        return bool(cv2.imwrite(path, title_image(text, WATERMARK_PX, WATERMARK_ALPHA)))
    except Exception:
        logger.exception('The watermark could not be made')
        return False


def build_share(clip: str, track: str | None, offset: float, watermark: str | None, encoder: video.Encoder, out: str,
                sample_rate: int = SAMPLE_RATE) -> None:
    """Write the share clip `out` from the players' clip: `track` from `offset` s (the point the room heard
    when the grin started) on the tape, or silence without one; the `watermark` PNG (or None) in the bottom
    right. Raises when it fails."""
    frames, fps, width, height = video.probe(clip)
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f'{clip} does not say its length')
    duration: float = frames / fps
    seconds: float = duration * SOURCE_MARGIN + 1.0
    if track:
        audio: np.ndarray = video.read_audio(track, offset, seconds, sample_rate)
    else:
        audio = np.zeros((round(seconds * sample_rate), 2), np.float32)
    pcm: np.ndarray = resample(audio, speed_curve(duration, sample_rate), sample_rate)
    overlays: list[tuple[str, str]] = [(watermark, 'W-w-24:H-h-24')] if watermark else []
    video.make_share(clip, overlays, pcm, sample_rate, encoder, out)
