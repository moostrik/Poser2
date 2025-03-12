
from enum import Enum

class PreviewType(Enum):
    NONE =  0
    VIDEO = 1
    MONO =  2
    STEREO= 3
    MASK =  4
    MASKED= 5

PreviewTypeNames: list[str] = [e.name for e in PreviewType]

exposureRange:          tuple[int, int] = (1000, 33000)
isoRange:               tuple[int, int] = ( 100, 1600 )
balanceRange:           tuple[int, int] = (1000, 12000)
contrastRange:          tuple[int, int] = ( -10, 10   )
brightnessRange:        tuple[int, int] = ( -10, 10   )
lumaDenoiseRange:       tuple[int, int] = (   0, 4    )
saturationRange:        tuple[int, int] = ( -10, 10   )
sharpnessRange:         tuple[int, int] = (   0, 4    )

stereoDepthRange:       tuple[int, int] = ( 500, 15000)
stereoBrightnessRange:  tuple[int, int] = (   0, 255  )

class StereoMedianFilterType(Enum):
    OFF = 0
    KERNEL_3x3 = 1
    KERNEL_5x5 = 2
    KERNEL_7x7 = 3

StereoMedianFilterTypeNames: list[str] = [e.name for e in StereoMedianFilterType]