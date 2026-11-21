"""The share clip: the visitor's grin as the phone gets it to keep and share. The clip player's clip (silent,
`grins/<style>/<id>.mp4`) with the visitor's track as its sound, tailored to the clip's length by the
tape (`audio/tape.py`: spun up over the first half second, wound down over the last) and a small
watermark in the bottom right, "SMILE", a circled "ai" (the machine's mark) and "by" the artist's name; no
caption, the joke is the room's. Built by the saver before the clip player's clip lands, the phone comes first."""

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

# the share clip for a phone: lighter than the players' clip, it travels
_SHARE_QUALITY: dict[video.Hardware, list[str]] = {
    video.Hardware.iGPU: ['-global_quality', '27'],
    video.Hardware.GPU:  ['-preset', 'p4', '-rc', 'vbr', '-cq', '27', '-b:v', '0'],
    video.Hardware.CPU:  ['-crf', '25', '-preset', 'veryfast', '-threads', '4'],
}


def write_watermark(text: str, path: str) -> bool:
    """The watermark, the machine's title with "by" the artist's name, as a PNG file for ffmpeg; False when it
    could not be made."""
    try:
        return bool(cv2.imwrite(path, title_image(text, WATERMARK_PX, WATERMARK_ALPHA)))
    except Exception:
        logger.exception('The watermark could not be made')
        return False


def build_share(clip: str, track: str | None, offset: float, watermark: str | None, encoder: video.Hardware, out: str,
                sample_rate: int = SAMPLE_RATE) -> None:
    """Write the share clip `out` from the clip player's clip: `track` from `offset` s (the point the room heard
    when the grin started) on the tape, or silence without one; the `watermark` PNG (or None) in the bottom
    right. Raises when it fails."""
    info = video.probe(clip)
    frames, fps, width, height = info.frames, info.fps, info.width, info.height
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
    make_share(clip, overlays, pcm, sample_rate, encoder, out)


def make_share(clip: str, overlays: list[tuple[str, str]], pcm: np.ndarray, rate: int, hardware: video.Hardware,
               out: str) -> None:
    """Write `out`: the clip re-encoded with the `overlays` (PNGs with alpha, each with its place as ffmpeg's
    overlay takes it, e.g. 'W-w-24:H-h-24') and `pcm` ((n, 2) float32 at `rate`) as its sound, cut to the
    shorter of the two; the clip's comment tag is kept. Raises when ffmpeg fails."""
    codec = video.encoder_name(video.ClipCodec.H264, hardware)
    _, pix_fmt = video.H264_OPTIONS[hardware]
    quality: list[str] = _SHARE_QUALITY[hardware]
    comment: str | None = video.probe(clip).comment
    metadata: list[str] = ['-metadata', f'comment={comment}'] if comment else []
    inputs: list[str] = ['-i', clip]
    chain: list[str] = []
    last: str = '[0:v]'
    for n, (png, place) in enumerate(overlays, start=1):
        inputs += ['-i', png]
        label: str = f'[v{n}]' if n < len(overlays) else ''
        chain.append(f'{last}[{n}:v]overlay={place}{label}')
        last = label
    filters: list[str] = ['-filter_complex', ';'.join(chain)] if chain else []
    audio_index: int = len(overlays) + 1
    args = ['-y', *inputs, '-f', 'f32le', '-ar', str(rate), '-ac', '2', '-i', 'pipe:0', *filters,
            '-map', f'{audio_index}:a:0', *([] if overlays else ['-map', '0:v:0']),
            '-c:v', codec, *quality, '-pix_fmt', pix_fmt, '-c:a', 'aac', '-b:a', '128k',
            '-shortest', '-movflags', '+faststart', *metadata, out]
    try:
        video.run(args, np.ascontiguousarray(pcm, dtype=np.float32).tobytes(), timeout=120)
    except RuntimeError as e:
        raise RuntimeError(f'could not make the share clip of {clip}: {e}') from e
