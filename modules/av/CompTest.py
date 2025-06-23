from modules.av.Definitions import *
from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods
from typing import Optional
import math
import time

class TestPattern(IntEnum):
    FILL = 0
    PULSE = auto()
    CHASE = auto()
    LINES = auto()
    RNDOM = auto()

TEST_PATTERN_NAMES: list[str] = [p.name for p in TestPattern]

methods_path: str = 'modules/av/CompTestMethods.py'

class TestParameters():
    def __init__(self) -> None:
        self.strength: float =  0.5
        self.speed: float =     0.5
        self.phase: float =     0.0
        self.width: float =     0.5
        self.amount: int =      36

class CompTest():
    def __init__(self, resolution: int) -> None:
        self.resolution: int = resolution
        self.pattern: TestPattern = TestPattern.FILL

        self.WP: TestParameters = TestParameters()
        self.BP: TestParameters = TestParameters()

        self.hot_reloader = HotReloadStaticMethods(self.__class__, True)

    def make_pattern(self) -> np.ndarray:
        try:
            white: Optional[np.ndarray] = None
            blue: Optional[np.ndarray] = None
            if self.pattern == TestPattern.FILL:
                white = self.make_fill(self.resolution, self.WP)
                blue = self.make_fill(self.resolution, self.BP)
            if self.pattern == TestPattern.PULSE:
                white = self.make_pulse(self.resolution, self.WP)
                blue = self.make_pulse(self.resolution, self.BP)
            if self.pattern == TestPattern.CHASE:
                white = self.make_chase(self.resolution, self.WP)
                blue = self.make_chase(self.resolution, self.BP)
            if self.pattern == TestPattern.LINES:
                white = self.make_lines(self.resolution, self.WP)
                blue = self.make_lines(self.resolution, self.BP)
            if self.pattern == TestPattern.RNDOM:
                white = self.make_random(self.resolution, self.WP)
                blue = self.make_random(self.resolution, self.BP)
            if white is not None and blue is not None:
                img: np.ndarray = np.zeros((1, self.resolution, 3), dtype=np.float16)
                img[0, :, 0] = white[0, :]
                img[0, :, 1] = blue[0, :]
                return img
        except Exception as e:
            print(f"[CompTest] Error generating pattern: {e}")

        return np.zeros((1, self.resolution, 3), dtype=np.float16)

    def set_pattern(self, pattern: TestPattern | int | str) -> None:
        if isinstance(pattern, str) and pattern in TEST_PATTERN_NAMES:
            self.pattern = TestPattern(TEST_PATTERN_NAMES.index(pattern))
        else:
            self.pattern = TestPattern(pattern)

    def reset(self) -> None:
        self.WP = TestParameters()
        self.BP = TestParameters()
        self.BP.phase = 0.5

    def white_to_blue(self) -> None:
        self.BP.strength =  self.WP.strength
        self.BP.speed =     self.WP.speed
        # self.BP.phase =     self.WP.phase
        self.BP.amount =    self.WP.amount
        self.BP.width =     self.WP.width

    def set_white_strength(self, value: float) -> None:
        self.WP.strength = max(0.0, min(1.0, value))
    def set_white_speed(self, value: float) -> None:
        self.WP.speed = min(1.0, max(-1.0, value))
    def set_white_phase(self, value: float) -> None:
        self.WP.phase = max(0.0, min(1.0, value))
    def set_white_width(self, value: float) -> None:
        self.WP.width = max(0.0, min(1.0, value))
    def set_white_amount(self, value: int) -> None:
        self.WP.amount = max(1, int(value))

    def set_blue_strength(self, value: float) -> None:
        self.BP.strength = max(0.0, min(1.0, value))
    def set_blue_speed(self, value: float) -> None:
        self.BP.speed = min(1.0, max(-1.0, value))
    def set_blue_phase(self, value: float) -> None:
        self.BP.phase = max(0.0, min(1.0, value))
    def set_blue_width(self, value: float) -> None:
        self.BP.width = max(0.0, min(1.0, value))
    def set_blue_amount(self, value: int) -> None:
        self.BP.amount = max(1, int(value))

    @staticmethod
    def make_fill(resolution: int, P: TestParameters) -> np.ndarray:
        return np.full((1, resolution), P.strength, dtype=np.float16) +0


    @staticmethod
    def make_pulse(resolution: int, P: TestParameters) -> np.ndarray:
        def lfo(time_: float, Htz: float, phase: float) -> float:
            return 0.5 * math.sin(time_ * math.pi * Htz + phase) + 0.5

        T: float = time.time()
        img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
        freq: float = P.speed
        img[0, :] = lfo(T, freq, P.phase * 2 * math.pi) * P.strength
        return img

    @staticmethod
    def make_chase(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
        if resolution == 0 or P.amount == 0:
            return img

        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            img[0, i] = value * P.strength
        return img

    @staticmethod
    def make_lines(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
        if resolution == 0 or P.amount == 0:
            return img

        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            value = 1.0 if value < P.width else 0.0
            img[0, i] = value * P.strength
        return img

    @staticmethod
    def make_random(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
        if resolution == 0:
            return img
        T: float = time.time() * P.speed
        for i in range(resolution):
            if math.sin(T + i) > 0.5:
                img[0, i] = P.strength
        return img

    @staticmethod
    def lfo(time: float, Htz: float, phase: float) -> float:
        return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

    # @staticmethod
    # def lfor(time: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    #     return CompTest.lfo(time, Htz, phase) * (rangeMax - rangeMin) + rangeMin
