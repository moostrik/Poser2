import math
import time
import numpy as np
from enum import IntEnum

from apps.white_space.composition.output import COMP_DTYPE
from modules.settings import BaseSettings, Field, Group
from modules.utils.HotReloadMethods import HotReloadMethods

import logging
logger = logging.getLogger(__name__)


class TestPattern(IntEnum):
    NONE  = 0
    FILL  = 1
    PULSE = 2
    CHASE = 3
    LINES = 4
    RNDOM = 5


TEST_PATTERN_NAMES: list[str] = [p.name for p in TestPattern]

IMG_MP = 1.0


class TestChannelSettings(BaseSettings):
    """Per-channel parameters for a test pattern (white or blue)."""
    strength: Field[float] = Field(0.5,  min=0.0,  max=1.0,  step=0.01, description="Brightness")
    speed:    Field[float] = Field(0.5,  min=-1.0, max=1.0,  step=0.01, description="Animation speed")
    phase:    Field[float] = Field(0.0,  min=0.0,  max=1.0,  step=0.01, description="Phase offset")
    width:    Field[float] = Field(0.5,  min=0.0,  max=1.0,  step=0.01, description="Pattern width")
    amount:   Field[int]   = Field(36,   min=1,    max=200,  step=1,    description="Pattern count")


class TestCompositionSettings(BaseSettings):
    """Settings for the test pattern compositor."""
    pattern: Field[TestPattern]          = Field(TestPattern.NONE, description="Active test pattern")
    white:   Group[TestChannelSettings]  = Group(TestChannelSettings)
    blue:    Group[TestChannelSettings]  = Group(TestChannelSettings)


class TestComposition:
    """Test pattern generator — activated when TestCompositionSettings.pattern != NONE."""

    def __init__(self, resolution: int, config: TestCompositionSettings) -> None:
        self.resolution: int = resolution
        self._config = config

        self.white_array: np.ndarray = np.zeros((1, resolution), dtype=COMP_DTYPE)
        self.blue_array:  np.ndarray = np.zeros((1, resolution), dtype=COMP_DTYPE)
        self.output_img:  np.ndarray = np.zeros((1, resolution, 3), dtype=COMP_DTYPE)
        self.indices:     np.ndarray = np.arange(resolution)

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def update(self) -> np.ndarray:
        W = self._config.white
        B = self._config.blue
        pattern = self._config.pattern
        try:
            if pattern == TestPattern.FILL:
                self.make_fill(self.white_array, W)
                self.make_fill(self.blue_array, B)
            elif pattern == TestPattern.PULSE:
                self.make_pulse(self.white_array, W)
                self.make_pulse(self.blue_array, B)
            elif pattern == TestPattern.CHASE:
                self.make_chase(self.white_array, W, self.indices)
                self.make_chase(self.blue_array, B, self.indices)
            elif pattern == TestPattern.LINES:
                self.make_lines(self.white_array, W, self.indices)
                self.make_lines(self.blue_array, B, self.indices)
            elif pattern == TestPattern.RNDOM:
                self.make_random(self.white_array, W, self.indices)
                self.make_random(self.blue_array, B, self.indices)

            self.output_img[0, :, 0] = self.white_array[0, :]
            self.output_img[0, :, 1] = self.blue_array[0, :]
        except Exception as e:
            logger.error(f"[TestComposition] Error generating pattern: {e}")
            self.output_img.fill(0)

        return self.output_img

    def reset(self) -> None:
        W = self._config.white; B = self._config.blue
        W.strength = 0.5; W.speed = 0.5; W.phase = 0.0; W.width = 0.5; W.amount = 36
        B.strength = 0.5; B.speed = 0.5; B.phase = 0.5; B.width = 0.5; B.amount = 36

    def white_to_blue(self) -> None:
        W = self._config.white; B = self._config.blue
        B.strength = W.strength
        B.speed    = W.speed
        B.width    = W.width
        B.amount   = W.amount

    @staticmethod
    def make_fill(array: np.ndarray, P: TestChannelSettings) -> None:
        array.fill(P.strength * IMG_MP)

    @staticmethod
    def make_pulse(array: np.ndarray, P: TestChannelSettings) -> None:
        T: float = time.time()
        value: float = (0.5 * math.sin(T * math.pi * P.speed + P.phase) + 0.5) * P.strength * IMG_MP
        array.fill(value)

    @staticmethod
    def make_chase(array: np.ndarray, P: TestChannelSettings, indices: np.ndarray) -> None:
        resolution: int = array.shape[1]
        adjusted_speed: float       = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float          = time.time() * adjusted_speed * 2 * math.pi
        phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
        array[0, :] = (0.5 * np.sin(phases) + 0.5) * P.strength * IMG_MP

    @staticmethod
    def make_lines(array: np.ndarray, P: TestChannelSettings, indices: np.ndarray) -> None:
        resolution: int = array.shape[1]
        adjusted_speed: float       = P.speed * P.amount / 10.0
        wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
        time_offset: float          = time.time() * adjusted_speed * 2 * math.pi
        phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
        values = 0.5 * np.sin(phases) + 0.5
        array[0, :] = np.where(values < P.width, P.strength * IMG_MP, 0.0)

    @staticmethod
    def make_random(array: np.ndarray, P: TestChannelSettings, indices: np.ndarray) -> None:
        T: float = time.time() * P.speed
        sine_values: np.ndarray = np.sin(T + indices)
        array[0, :] = np.where(sine_values > 0.5, P.strength * IMG_MP, 0)
