"""The share clip: the visitor's grin as the phone gets it to keep and share. The players' clip (silent,
`grins/<style>/<id>.mp4`) with the visitor's track as its sound, tailored to the clip's length by the
tape (modules/audio/tape.py: spun up over the first half second, wound down over the last) and a small
watermark in the bottom right, "SMILE", a circled "ai" (the machine's mark) and "by" the artist's name; no
caption, the joke is the room's. Built by the saver before the players' clip lands, the phone comes first."""

import logging

import cv2
import freetype
import numpy as np

from ..audio import resample, speed_curve
from ..credit_scroll import FONT, _render, _render_baseline
from modules import video

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 48000
WATERMARK_PX: int = 28          # the watermark's ink height, on the 896 x 1120 clip
WATERMARK_ALPHA: float = 0.8
SOURCE_MARGIN: float = 1.4      # the track read is this much longer than the clip: the spin-up runs at up to 1.5x
# the mark after SMILE, in fractions of the text's font size, as the page's title draws it (relay/web/style.css,
# `.ai` and `#title .ai`: the ring 1.5 em of letters at 0.5 em, its border 0.14 em of them, raised 0.9 em of them)
MARK_RING: float = 0.75         # the ring's outer diameter
MARK_STROKE: float = 0.07       # the ring's thickness
MARK_LETTERS: float = 0.5       # the font size of "ai"
MARK_LIFT: float = 0.635        # the ring's center above the baseline (the letters' baseline 0.45 up, their ink 0.37 tall)
MARK_GAP: float = 0.2           # from the E to the ring


def watermark_image(text: str, height_px: int = WATERMARK_PX, alpha: float = WATERMARK_ALPHA, scale: int = 4) -> np.ndarray:
    """The watermark as a BGRA image: "SMILE" with the machine's mark after it, the ring with "ai" in it as the
    page's title has it (smaller than the letters and raised above them like a (R), MARK_*), and "by" the
    artist's name after that (`text`; "SMILE" and the mark alone without one), all on one baseline in the
    credit's font, white with one black edge round all of it (drawn at `scale` times the size, the edge grown
    like the screen's text, then scaled down)."""
    face = freetype.Face(FONT)
    size: int = height_px * scale
    font: int = round(size * 0.9)       # the text's font size; its capitals are 0.74 of it
    title, title_base = _render_baseline(face, 'SMILE', font)
    name, name_base = _render_baseline(face, f'by {text.strip()}', font) if text.strip() else (np.zeros((0, 0), np.uint8), 0)
    letters: np.ndarray = _render(face, 'ai', round(MARK_LETTERS * font))
    ring: int = round(MARK_RING * font)
    stroke: int = max(2, round(MARK_STROKE * font))
    lift: int = round(MARK_LIFT * font)
    gap: int = round(MARK_GAP * font)
    after: int = round(0.3 * font)
    # the rows: the baseline low enough for the raised ring, and under it the name's descenders
    baseline: int = max(title_base, lift + ring // 2 + 1)
    descent: int = max(1, title.shape[0] - title_base, name.shape[0] - name_base if name.size else 0)
    width: int = title.shape[1] + gap + ring + (after + name.shape[1] if name.size else 0)
    ink: np.ndarray = np.zeros((baseline + descent, width), np.uint8)

    def put(piece: np.ndarray, base: int, x: int) -> None:
        h, w = piece.shape[:2]
        y: int = baseline - base
        np.maximum(ink[y:y + h, x:x + w], piece, out=ink[y:y + h, x:x + w])

    put(title, title_base, 0)
    center: tuple[int, int] = (title.shape[1] + gap + ring // 2, baseline - lift)
    cv2.circle(ink, center, (ring - stroke) // 2, 255, stroke, cv2.LINE_AA)
    h, w = letters.shape[:2]
    y0: int = center[1] - h // 2       # the letters' ink centered in the ring
    x0: int = center[0] - w // 2
    np.maximum(ink[y0:y0 + h, x0:x0 + w], letters, out=ink[y0:y0 + h, x0:x0 + w])
    if name.size:
        put(name, name_base, title.shape[1] + gap + ring + after)
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
