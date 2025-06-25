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

        self.hot_reloader = HotReloadStaticMethods(self.__class__, True)

    def make_pattern(self) -> np.ndarray:
        try:
            white_filled = False
            blue_filled = False

            if self.pattern == TestPattern.FILL:
                self.fill_array(self.white_array, self.WP)
                self.fill_array(self.blue_array, self.BP)
                white_filled = blue_filled = True
            elif self.pattern == TestPattern.PULSE:
                self.pulse_array(self.white_array, self.WP)
                self.pulse_array(self.blue_array, self.BP)
                white_filled = blue_filled = True
            elif self.pattern == TestPattern.CHASE:
                self.chase_array(self.white_array, self.WP)
                self.chase_array(self.blue_array, self.BP)
                white_filled = blue_filled = True
            elif self.pattern == TestPattern.LINES:
                self.lines_array(self.white_array, self.WP)
                self.lines_array(self.blue_array, self.BP)
                white_filled = blue_filled = True
            elif self.pattern == TestPattern.RNDOM:
                self.random_array(self.white_array, self.WP)
                self.random_array(self.blue_array, self.BP)
                white_filled = blue_filled = True

            if white_filled and blue_filled:
                # Copy data to output array
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
        return np.full((1, resolution), P.strength, dtype=IMG_TYPE)


    @staticmethod
    def make_pulse(resolution: int, P: TestParameters) -> np.ndarray:
        def lfo(time_: float, Htz: float, phase: float) -> float:
            return 0.5 * math.sin(time_ * math.pi * Htz + phase) + 0.5

        T: float = time.time()
        img: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        freq: float = P.speed
        img[0, :] = lfo(T, freq, P.phase * 2 * math.pi) * P.strength * IMG_MP
        return img

    @staticmethod
    def make_chase(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        if resolution == 0 or P.amount == 0:
            return img

        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            img[0, i] = value * P.strength * IMG_MP
        return img

    @staticmethod
    def make_lines(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        if resolution == 0 or P.amount == 0:
            return img

        adjusted_speed: float = P.speed * P.amount / 10.0 * 8
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            value = 1.0 if value < P.width else 0.0
            img[0, i] = value * P.strength * IMG_MP
        return img

    @staticmethod
    def make_random(resolution: int, P: TestParameters) -> np.ndarray:
        img: np.ndarray = np.zeros((1, resolution), dtype=IMG_TYPE)
        if resolution == 0:
            return img
        T: float = time.time() * P.speed
        for i in range(resolution):
            if math.sin(T + i) > 0.5:
                img[0, i] = P.strength  * IMG_MP
        return img

    @staticmethod
    def lfo(time: float, Htz: float, phase: float) -> float:
        return 0.5 * math.sin(time * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

    def fill_array(self, array: np.ndarray, P: TestParameters) -> None:
        array.fill(P.strength * IMG_MP)

    def pulse_array(self, array: np.ndarray, P: TestParameters) -> None:
        T: float = time.time()
        freq: float = P.speed
        value = self.lfo(T, freq, P.phase) * P.strength * IMG_MP
        array.fill(value)

    def chase_array(self, array: np.ndarray, P: TestParameters) -> None:
        if self.resolution == 0 or P.amount == 0:
            array.fill(0)
            return

        adjusted_speed: float = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / self.resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(self.resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            array[0, i] = value * P.strength * IMG_MP

    def lines_array(self, array: np.ndarray, P: TestParameters) -> None:
        if self.resolution == 0 or P.amount == 0:
            array.fill(0)
            return

        adjusted_speed: float = P.speed * P.amount / 10.0 * 8
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / self.resolution
        time_offset: float = time.time() * adjusted_speed * 2 * math.pi

        for i in range(self.resolution):
            phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
            value: float = 0.5 * math.sin(phase) + 0.5
            value = 1.0 if value < P.width else 0.0
            array[0, i] = value * P.strength * IMG_MP

    def random_array(self, array: np.ndarray, P: TestParameters) -> None:
        if self.resolution == 0:
            array.fill(0)
            return
            
        T: float = time.time() * P.speed
        for i in range(self.resolution):
            array[0, i] = P.strength * IMG_MP if math.sin(T + i) > 0.5 else 0
