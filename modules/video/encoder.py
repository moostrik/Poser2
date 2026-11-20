"""The h264 encoders a clip can be written with, and the trial that picks one that works."""

import logging
import subprocess
from enum import Enum

from .process import available, error_text, popen

logger = logging.getLogger(__name__)


class Encoder(Enum):
    AUTO  = 'Auto'
    QSV   = 'Intel Quick Sync'
    NVENC = 'Nvidia NVENC'
    CPU   = 'CPU (libx264)'


# codec, quality options, pixel format the codec takes
CODECS: dict[Encoder, tuple[str, list[str], str]] = {
    Encoder.QSV:   ('h264_qsv',   ['-global_quality', '20'], 'nv12'),
    Encoder.NVENC: ('h264_nvenc', ['-preset', 'p4', '-rc', 'vbr', '-cq', '20', '-b:v', '0'], 'yuv420p'),
    Encoder.CPU:   ('libx264',    ['-crf', '18', '-preset', 'veryfast', '-threads', '4'], 'yuv420p'),
}
_AUTO_ORDER: list[Encoder] = [Encoder.QSV, Encoder.NVENC, Encoder.CPU]


TRIAL_SIZE: int = 256      # px, the trial frame's side; NVENC refuses frames 128 wide and less


def encoder_works(encoder: Encoder) -> bool:
    """A one-frame trial encode: the encoder list alone does not prove the driver works."""
    codec, quality, pix_fmt = CODECS[encoder]
    args = ['-nostdin', '-f', 'lavfi', '-i', f'color=c=gray:size={TRIAL_SIZE}x{TRIAL_SIZE}:rate=15', '-frames:v', '1',
            '-c:v', codec, *quality, '-pix_fmt', pix_fmt, '-f', 'null', '-']
    try:
        process = popen(args, subprocess.DEVNULL, subprocess.DEVNULL)
        _, err = process.communicate(timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning('%s trial failed: %s', encoder.value, e)
        return False
    if process.returncode != 0:
        logger.info('%s not usable: %s', encoder.value, error_text(err))
        return False
    return True


def pick_encoder(preference: Encoder = Encoder.AUTO) -> Encoder:
    """The encoder to use: the preferred one if its trial passes, else the first of Intel, Nvidia, CPU that does."""
    if not available():
        logger.error('ffmpeg or ffprobe not on the PATH: clips cannot be saved')
        return Encoder.CPU
    candidates = _AUTO_ORDER if preference == Encoder.AUTO else [preference, Encoder.CPU]
    for encoder in candidates:
        if encoder_works(encoder):
            if preference not in (Encoder.AUTO, encoder):
                logger.warning('%s not usable, using %s', preference.value, encoder.value)
            return encoder
    logger.error('No h264 encoder works, using %s anyway', Encoder.CPU.value)
    return Encoder.CPU
