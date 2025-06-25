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

        # Pre-allocate arrays
        self.white_array: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        self.blue_array: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        self.output_img: np.ndarray = np.zeros((1, resolution, 3), dtype=IMG_TYPE)
        self.indices: np.ndarray = np.arange(resolution)

        self.hot_reloader = HotReloadStaticMethods(self.__class__, True)

    def make_pattern(self) -> np.ndarray:
        try:
            if self.pattern == TestPattern.FILL:
                self.make_fill(self.white_array, self.WP)
                self.make_fill(self.blue_array, self.BP)
            elif self.pattern == TestPattern.PULSE:
                self.make_pulse(self.white_array, self.WP)
                self.make_pulse(self.blue_array, self.BP)
            elif self.pattern == TestPattern.CHASE:
                self.make_chase(self.white_array, self.WP, self.indices)
                self.make_chase(self.blue_array, self.BP, self.indices)
            elif self.pattern == TestPattern.LINES:
                self.make_lines(self.white_array, self.WP, self.indices)
                self.make_lines(self.blue_array, self.BP, self.indices)
            elif self.pattern == TestPattern.RNDOM:
                self.make_random(self.white_array, self.WP, self.indices)
                self.make_random(self.blue_array, self.BP, self.indices)

            self.output_img[0, :, 0] = self.white_array[0, :]
            self.output_img[0, :, 1] = self.blue_array[0, :]
        except Exception as e:
            print(f"[CompTest] Error generating pattern: {e}")
            # Clear output image to be safe
            self.output_img.fill(0)

        return self.output_img

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
        self.BP.width =     self.WP.width
        self.BP.amount =    self.WP.amount

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
    def make_fill(array: np.ndarray, P: TestParameters) -> None:
        array.fill(P.strength * IMG_MP)

    @staticmethod
    def make_pulse(array: np.ndarray, P: TestParameters) -> None:
        T: float = time.time()
        phase_angle = T * math.pi * P.speed + P.phase
        value: float = (0.5 * math.sin(phase_angle) + 0.5) * P.strength * IMG_MP
        array.fill(value)

    @staticmethod
    def make_chase(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
        resolution: int = array.shape[1]
        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        # Vectorized
        phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
        array[0, :] = (0.5 * np.sin(phases) + 0.5) * P.strength * IMG_MP

        # # Old version for reference
        # for i in range(resolution):
        #     phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
        #     value: float = 0.5 * math.sin(phase) + 0.5
        #     array[0, i] = value * P.strength * IMG_MP

    @staticmethod
    def make_lines(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
        resolution: int = array.shape[1]
        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        # Vectorized
        phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
        values = 0.5 * np.sin(phases) + 0.5
        array[0, :] = np.where(values < P.width, P.strength * IMG_MP, 0.0)

        # # Old version for reference
        # for i in range(resolution):
        #     phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
        #     value: float = 0.5 * math.sin(phase) + 0.5
        #     value = 1.0 if value < P.width else 0.0
        #     array[0, i] = value * P.strength * IMG_MP

    @staticmethod
    def make_random(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
        T: float = time.time() * P.speed
        sine_values: np.ndarray = np.sin(T + indices)
        array[0, :] = np.where(sine_values > 0.5, P.strength * IMG_MP, 0)

