from modules.av.Definitions import *
from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods
from typing import Optional

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
        self.amount: int =      36
        self.width: int =       3

class CompTest():
    def __init__(self, resolution: int) -> None:
        self.resolution: int = resolution
        self.pattern: TestPattern = TestPattern.FILL

        self.WP: TestParameters = TestParameters()
        self.BP: TestParameters = TestParameters()

        self.hot_reloader = HotReloadStaticMethods(self.__class__, methods_path)


    def make_pattern(self) -> np.ndarray:
        try:
            white: Optional[np.ndarray] = None
            blue: Optional[np.ndarray] = None
            if self.pattern == TestPattern.FILL:
                white = self.make_fill(self.resolution, self.WP.strength)
                blue = self.make_fill(self.resolution, self.BP.strength)
            if self.pattern == TestPattern.PULSE:
                white = self.make_pulse(self.resolution, self.WP.strength, self.WP.speed, self.WP.phase)
                blue = self.make_pulse(self.resolution, self.BP.strength, self.BP.speed, self.BP.phase)
            if self.pattern == TestPattern.CHASE:
                white = self.make_chase(self.resolution, self.WP.strength, self.WP.speed, self.WP.phase, self.WP.amount)
                blue = self.make_chase(self.resolution, self.BP.strength, self.BP.speed, self.BP.phase, self.BP.amount)
            if self.pattern == TestPattern.LINES:
                white = self.make_lines(self.resolution, self.WP.strength, self.WP.speed, self.WP.phase, self.WP.amount, self.WP.width)
                blue = self.make_lines(self.resolution, self.BP.strength, self.BP.speed, self.BP.phase, self.BP.amount, self.BP.width)
            if self.pattern == TestPattern.RNDOM:
                white = self.make_random(self.resolution, self.WP.strength, self.WP.speed)
                blue = self.make_random(self.resolution, self.BP.strength, self.BP.speed)
            if white is not None and blue is not None:
                img: np.ndarray = np.zeros((1, self.resolution, 3), dtype=np.float16)
                img[0, :, 0] = white[0, :]  # White channel
                img[0, :, 1] = blue[0, :]   # Blue channel
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
    def set_white_amount(self, value: int) -> None:
        self.WP.amount = max(1, int(value))
    def set_white_width(self, value: int) -> None:
        self.WP.width = max(1, int(value))

    def set_blue_strength(self, value: float) -> None:
        self.BP.strength = max(0.0, min(1.0, value))
    def set_blue_speed(self, value: float) -> None:
        self.BP.speed = min(1.0, max(-1.0, value))
    def set_blue_phase(self, value: float) -> None:
        self.BP.phase = max(0.0, min(1.0, value))
    def set_blue_amount(self, value: int) -> None:
        self.BP.amount = max(1, int(value))
    def set_blue_width(self, value: int) -> None:
        self.BP.width = max(1, int(value))

    @staticmethod
    def make_fill(resolution: int, strength: float) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution), dtype=np.float16)

    @staticmethod
    def make_pulse(resolution: int, strength: float, speed: float, phase: float) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution), dtype=np.float16)

    @staticmethod
    def make_chase(resolution: int, strength: float, speed: float, phase: float, amount: int) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution), dtype=np.float16)

    @staticmethod
    def make_lines(resolution: int, strength: float, speed: float, phase: float, amount: int, width: int) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution), dtype=np.float16)

    @staticmethod
    def make_random(resolution: int, strength: float, speed: float) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution), dtype=np.float16)

    # @staticmethod
    # def lfo(time: float, Htz: float, phase: float) -> float:
    #     return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

    # @staticmethod
    # def lfor(time: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    #     return CompTest.lfo(time, Htz, phase) * (rangeMax - rangeMin) + rangeMin
