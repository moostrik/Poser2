import math


def lfo(timestamp: float, frequency: float, phase: float = 0) -> float:
    """A sine between 0 and 1 at `timestamp` (s), `frequency` (Hz) and `phase` (radians)."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * frequency * timestamp + phase)
