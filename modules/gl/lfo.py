import math
from time import perf_counter


def lfo(frequency: float, phase: float = 0) -> float:
    """A sine between 0 and 1 at `frequency` (Hz) and `phase` (radians), on perf_counter."""
    elapsed_time: float = perf_counter()
    lfo_value: float = 0.5 + 0.5 * math.sin(2 * math.pi * frequency * elapsed_time + phase)
    return lfo_value
