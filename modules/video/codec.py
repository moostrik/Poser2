"""The codec choices of the camera recordings: which encoder and which container, and what ffmpeg
calls them."""

from enum import Enum


class CoderType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2


class CoderFormat(Enum):
    H264 = '.mp4'
    H265 = '.hevc'


ENCODER_NAMES: dict[CoderFormat, dict[CoderType, str]] = {
    CoderFormat.H264: {
        CoderType.CPU:  'libx264',
        CoderType.GPU:  'h264_nvenc',
        CoderType.iGPU: 'h264_qsv'
    },
    CoderFormat.H265: {
        CoderType.CPU:  'libx265',
        CoderType.GPU:  'hevc_nvenc',
        CoderType.iGPU: 'hevc_qsv'
    }
}

# ffmpeg's hwaccel name and device index per decoder: the CPU decodes without either.
HWACCEL_NAMES: dict[CoderType, str] = {
    CoderType.CPU:  '',
    CoderType.GPU:  'd3d12va',
    CoderType.iGPU: 'd3d12va'
}

HWACCEL_DEVICES: dict[CoderType, str] = {
    CoderType.CPU:  '',
    CoderType.GPU:  '0',
    CoderType.iGPU: '1'
}


def encoder_name(video_format: CoderFormat, encoder: CoderType) -> str:
    """The ffmpeg `vcodec` for a format on an encoder."""
    return ENCODER_NAMES[video_format][encoder]


def hwaccel(decoder: CoderType) -> tuple[str, str]:
    """ffmpeg's `(hwaccel, hwaccel_device)` for a decoder; both empty for the CPU."""
    return HWACCEL_NAMES[decoder], HWACCEL_DEVICES[decoder]
