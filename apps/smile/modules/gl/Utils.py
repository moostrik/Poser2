from time import time
import math

class FpsCounter:
    def __init__(self, numSamples = 120) -> None:
        self._times: list[float] = []
        self.numSamples: int = numSamples

    def tick(self) -> None:
        now: float = time()
        self._times.append(now)
        if len(self._times) > self.numSamples:
            self._times.pop(0)

    def get_fps(self) -> int:
        if len(self._times) < 2:
            return 0
        diff: float = self._times[-1] - self._times[0]
        if diff == 0:
            return 0
        return int(math.floor(len(self._times) / diff))


def lfo(frequency, phase=0) -> float:
    elapsed_time: float = time()
    lfo_value: float = 0.5 + 0.5 * math.sin(2 * math.pi * frequency * elapsed_time + phase)
    return lfo_value

def fit(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> list[float]:
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio > src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return [x, y, width, height]

def fill(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> list[float]:
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio < src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return [x, y, width, height]
