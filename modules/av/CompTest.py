from typing import Any, Callable, Type
from modules.av.Definitions import *
import math
import time

from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods

class TestPattern(IntEnum):
    FILL = 0
    PULSE = auto()
    CHASE = auto()
    SNGLE = auto()
    RNDOM = auto()

TEST_PATTERN_NAMES: list[str] = [p.name for p in TestPattern]

method_types: dict[str, Any] = {
    "make_fill": Callable[[int, float, float], np.ndarray],
    "make_pulse": Callable[[int, float, float, float, float], np.ndarray],
    "make_chase": Callable[[int, float, float, float, float, int], np.ndarray],
    "make_single": Callable[[int, float, float, float, float, int], np.ndarray],
    "make_random": Callable[[int, float, float, float], np.ndarray]
}

methods_path: str = 'modules/av/CompTestMethods.py'

class CompTest():
    def __init__(self, resolution: int) -> None:
        self.resolution: int = resolution
        self.pattern: TestPattern = TestPattern.FILL

        self.white_strength: float = 1.0
        self.blue_strength: float = 0.0
        self.blue_phase: float = 0.0

        self.pulse_speed: float = 0.0
        self.chase_speed: float = 0.0
        self.chase_amount: int = 0

        self.single_speed: float = 0.0
        self.single_amount: int = 0
        self.random_speed: float = 0.0

        self.method_reloader: HotReloadStaticMethods = HotReloadStaticMethods(
            methods_file_path=methods_path,
            target_class=CompTest,
            method_types=method_types
        )

    def make_pattern(self) -> np.ndarray:
        try:
            if self.pattern == TestPattern.FILL:
                return self.make_fill(self.resolution, self.white_strength, self.blue_strength)
            if self.pattern == TestPattern.PULSE:
                return self.make_pulse(self.resolution, self.white_strength, self.blue_strength, self.blue_phase, self.pulse_speed)
            if self.pattern == TestPattern.CHASE:
                return self.make_chase(self.resolution, self.white_strength, self.blue_strength, self.blue_phase, self.chase_speed, self.chase_amount)
            if self.pattern == TestPattern.SNGLE:
                return self.make_single(self.resolution, self.white_strength, self.blue_strength, self.blue_phase, self.single_speed, self.single_amount)
            if self.pattern == TestPattern.RNDOM:
                return self.make_random(self.resolution, self.white_strength, self.blue_strength, self.random_speed)
        except Exception as e:
            print(f"[CompTest] Error generating pattern: {e}")

        return np.zeros((1, self.resolution, 3), dtype=np.float16)

    def set_pattern(self, pattern: TestPattern | int | str) -> None:
        if isinstance(pattern, str) and pattern in TEST_PATTERN_NAMES:
            self.pattern = TestPattern(TEST_PATTERN_NAMES.index(pattern))
        else:
            self.pattern = TestPattern(pattern)

    def set_white_strength(self, strength: float) -> None:
        self.white_strength = max(0.0, min(1.0, strength))

    def set_blue_strength(self, strength: float) -> None:
        self.blue_strength = max(0.0, min(1.0, strength))

    def set_blue_phase(self, phase: float) -> None:
        self.blue_phase = max(0.0, min(1.0, phase))

    def set_pulse_speed(self, speed: float) -> None:
        self.pulse_speed = min(1.0, max(-1.0, speed))

    def set_chase_speed(self, speed: float) -> None:
        self.chase_speed = min(1.0, max(-1.0, speed))

    def set_chase_amount(self, amount: int) -> None:
        self.chase_amount = max(1, amount)

    def set_single_speed(self, speed: float) -> None:
        self.single_speed = min(1.0, max(-1.0, speed))

    def set_single_amount(self, amount: int) -> None:
        self.single_amount = max(1, amount)

    def set_random_speed(self, speed: float) -> None:
        self.random_speed = max(0.0, speed)

    @staticmethod
    def make_fill(resolution: int, white_strength: float, blue_strength) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution, 3), dtype=np.float16)

    @staticmethod
    def make_pulse(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, pulse_speed: float) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution, 3), dtype=np.float16)

    @staticmethod
    def make_chase(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, chase_speed: float, chase_amount: int) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution, 3), dtype=np.float16)

    @staticmethod
    def make_single(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, single_speed: float, single_amount: int) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution, 3), dtype=np.float16)

    @staticmethod
    def make_random(resolution: int, white_strength: float, blue_strength: float, random_speed: float) -> np.ndarray:
        """Placeholder"""
        return np.zeros((1, resolution, 3), dtype=np.float16)

    # @staticmethod
    # def lfo(time: float, Htz: float, phase: float) -> float:
    #     return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

    # @staticmethod
    # def lfor(time: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    #     return CompTest.lfo(time, Htz, phase) * (rangeMax - rangeMin) + rangeMin
