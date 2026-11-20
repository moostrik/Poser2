"""The h264 encoder a clip is written with on each chip, and the trial that picks a chip whose encoder works."""

import logging
import subprocess
from .hardware import ClipCodec, Hardware, encoder_name
from .process import available, error_text, popen

logger = logging.getLogger(__name__)


# quality options and the pixel format each chip's h264 encoder takes
H264_OPTIONS: dict[Hardware, tuple[list[str], str]] = {
    Hardware.iGPU: (['-global_quality', '20'], 'nv12'),
    Hardware.GPU:  (['-preset', 'p4', '-rc', 'vbr', '-cq', '20', '-b:v', '0'], 'yuv420p'),
    Hardware.CPU:  (['-crf', '18', '-preset', 'veryfast', '-threads', '4'], 'yuv420p'),
}
_AUTO_ORDER: list[Hardware] = [Hardware.iGPU, Hardware.GPU, Hardware.CPU]


TRIAL_SIZE: int = 256      # px, the trial frame's side; NVENC refuses frames 128 wide and less


def encoder_works(hardware: Hardware) -> bool:
    """A one-frame trial h264 encode on `hardware`: the encoder list alone does not prove the driver works."""
    codec = encoder_name(ClipCodec.H264, hardware)
    quality, pix_fmt = H264_OPTIONS[hardware]
    args = ['-nostdin', '-f', 'lavfi', '-i', f'color=c=gray:size={TRIAL_SIZE}x{TRIAL_SIZE}:rate=15', '-frames:v', '1',
            '-c:v', codec, *quality, '-pix_fmt', pix_fmt, '-f', 'null', '-']
    try:
        process = popen(args, subprocess.DEVNULL, subprocess.DEVNULL)
        _, err = process.communicate(timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning('%s trial failed: %s', hardware.value, e)
        return False
    if process.returncode != 0:
        logger.info('%s not usable: %s', hardware.value, error_text(err))
        return False
    return True


def pick_encoder(preference: Hardware = Hardware.AUTO) -> Hardware:
    """The chip to encode on: the preferred one if its trial passes, else the first of the iGPU, the GPU, the CPU
    that does."""
    if not available():
        logger.error('ffmpeg or ffprobe not on the PATH: clips cannot be saved')
        return Hardware.CPU
    candidates = _AUTO_ORDER if preference == Hardware.AUTO else [preference, Hardware.CPU]
    for hardware in candidates:
        if encoder_works(hardware):
            if preference not in (Hardware.AUTO, hardware):
                logger.warning('%s not usable, using %s', preference.value, hardware.value)
            return hardware
    logger.error('No h264 encoder works, using %s anyway', Hardware.CPU.value)
    return Hardware.CPU
