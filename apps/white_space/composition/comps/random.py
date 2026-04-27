"""Random composition - 2D fBm noise: spatial x time, no directional scrolling."""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Group

from ..base import Composition
from ..transport import Transport


_TABLE_SIZE = 256
_MASK       = _TABLE_SIZE - 1


class RandomChannelSettings(BaseSettings):
    strength:    Field[float] = Field(0.5, min=0.0, max=1.0,  step=0.01, description="Output amplitude")
    speed:       Field[float] = Field(0.5, min=0.0, max=16.0, step=0.01, description="Temporal evolution rate (beats per noise period)")
    scale:       Field[float] = Field(4.0, min=0.5, max=32.0, step=0.1,  description="Spatial feature size (low=large blobs, high=fine grain)")
    octaves:     Field[int]   = Field(3,   min=1,   max=6,               description="fBm octave count (1=smooth, 6=detailed)")
    persistence: Field[float] = Field(0.5, min=0.1, max=1.0,  step=0.01, description="Octave amplitude decay")
    threshold:   Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, description="Hard gate: 0=smooth gradient, >0=lit pixels above this level only")


class RandomSettings(BaseSettings):
    enabled: Field[bool]  = Field(False, description="Enable Random composition")
    gain:    Field[float] = Field(1.0,   min=0.0, max=2.0, step=0.01, description="Output gain")
    white:   Group[RandomChannelSettings] = Group(RandomChannelSettings)
    blue:    Group[RandomChannelSettings] = Group(RandomChannelSettings)


class Random(Composition):
    """2D fBm value noise: noise(pixel * scale, beat_time * speed).
    Each pixel evolves independently over time - no directional motion.
    """

    def __init__(self, resolution: int, config: RandomSettings) -> None:
        super().__init__(resolution, config)
        self._config    = config
        self._positions = np.linspace(0.0, 1.0, resolution, endpoint=False, dtype=np.float64)

        rng = np.random.default_rng(seed=42)
        self._perm_w: np.ndarray = rng.permutation(_TABLE_SIZE).astype(np.int32)
        self._vals_w: np.ndarray = rng.random(_TABLE_SIZE)
        self._perm_b: np.ndarray = rng.permutation(_TABLE_SIZE).astype(np.int32)
        self._vals_b: np.ndarray = rng.random(_TABLE_SIZE)

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        if not self._config.enabled:
            return

        beat_time = transport.beat + transport.phase
        W = self._config.white
        B = self._config.blue
        g = self._config.gain

        white += self._fbm(self._perm_w, self._vals_w, self._positions,
                           beat_time, W.scale, W.speed, W.octaves,
                           W.persistence, W.threshold, W.strength * g)
        blue  += self._fbm(self._perm_b, self._vals_b, self._positions,
                           beat_time, B.scale, B.speed, B.octaves,
                           B.persistence, B.threshold, B.strength * g)

    @staticmethod
    def _fbm(
        perm:        np.ndarray,
        vals:        np.ndarray,
        positions:   np.ndarray,
        beat_time:   float,
        scale:       float,
        speed:       float,
        octaves:     int,
        persistence: float,
        threshold:   float,
        amplitude:   float,
    ) -> np.ndarray:
        result  = np.zeros(len(positions), dtype=np.float64)
        freq_x  = scale
        freq_t  = speed
        amp     = 1.0
        max_val = 0.0

        for _ in range(octaves):
            X  = positions * freq_x
            T  = beat_time * freq_t

            ix   = np.floor(X).astype(np.int32) & _MASK
            ix1  = (ix + 1) & _MASK
            iy   = int(math.floor(T)) & _MASK
            iy1  = (iy + 1) & _MASK
            fx   = X - np.floor(X)
            fy   = T - math.floor(T)
            fx_s = fx * fx * (3.0 - 2.0 * fx)
            fy_s = fy * fy * (3.0 - 2.0 * fy)

            v00 = vals[perm[(perm[ix ] + iy ) & _MASK]]
            v10 = vals[perm[(perm[ix1] + iy ) & _MASK]]
            v01 = vals[perm[(perm[ix ] + iy1) & _MASK]]
            v11 = vals[perm[(perm[ix1] + iy1) & _MASK]]

            v0      = v00 + fx_s * (v10 - v00)
            v1      = v01 + fx_s * (v11 - v01)
            result += (v0 + fy_s * (v1 - v0)) * amp

            max_val += amp
            amp     *= persistence
            freq_x  *= 2.0
            freq_t  *= 2.0

        result /= max_val

        if threshold > 0.0:
            result = np.where(result >= threshold, (result - threshold) / (1.0 - threshold + 1e-9), 0.0)

        return (result * amplitude).astype(np.float32)
