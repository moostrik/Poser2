"""The chips video is encoded and decoded on, the codecs, and what ffmpeg calls an encoder or a hardware decode on each."""

from enum import Enum


class Hardware(Enum):
    """The chip a clip is encoded or decoded on. AUTO, to encode: the first whose trial passes, the iGPU, the GPU, the
    CPU."""
    AUTO = 'Auto'
    CPU  = 'CPU'
    iGPU = 'Intel iGPU'
    GPU  = 'Nvidia GPU'


class ClipCodec(Enum):
    H264 = 'h264'
    HEVC = 'hevc'


ENCODER_NAMES: dict[tuple[ClipCodec, Hardware], str] = {
    (ClipCodec.H264, Hardware.CPU):  'libx264',
    (ClipCodec.H264, Hardware.iGPU): 'h264_qsv',
    (ClipCodec.H264, Hardware.GPU):  'h264_nvenc',
    (ClipCodec.HEVC, Hardware.CPU):  'libx265',
    (ClipCodec.HEVC, Hardware.iGPU): 'hevc_qsv',
    (ClipCodec.HEVC, Hardware.GPU):  'hevc_nvenc',
}

DECODE_API: str = 'd3d12va'     # on the iGPU 25 % faster than d3d11va for a recording's chunk
DECODE_ADAPTERS: dict[Hardware, int] = {Hardware.GPU: 0, Hardware.iGPU: 1}     # DXGI adapter indices on the rig


def encoder_name(codec: ClipCodec, hardware: Hardware) -> str:
    """What ffmpeg calls `codec`'s encoder on `hardware` (not AUTO)."""
    return ENCODER_NAMES[(codec, hardware)]


def hwaccel_args(hardware: Hardware) -> list[str]:
    """ffmpeg's input options for a hardware decode on `hardware`; none for the CPU (and AUTO), which decodes in
    software."""
    adapter = DECODE_ADAPTERS.get(hardware)
    if adapter is None:
        return []
    return ['-hwaccel', DECODE_API, '-hwaccel_device', str(adapter)]
