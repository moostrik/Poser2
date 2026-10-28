"""The tape: the music's speed as the Max patch plays it (`p speedVar` in data/audio/Smile-Music-Player.maxpat).
At the flash the track spins up from standstill along a curve with a wobble, the pitch following; at the
end it winds down to a stop. The phone page plays the live track with the same curve (relay/web/app.js,
SPIN_UP, from the room's stage progress); here the curve shapes the sound of the share clip, a loop of
its own length: the spin-up over its first half second, the wind-down over its last second."""

import numpy as np

# the patch's function, x the flash-in progress (0..1), y the speed before the gain
SPIN_UP: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.122, 0.333), (0.559, 0.573), (0.707, 1.0), (0.814, 0.653), (1.0, 0.773))
SPIN_UP_GAIN: float = 1.5       # the patch's `* 1.5`
SPIN_UP_FLOOR: float = 0.3      # a speed under this is standstill (the patch's `< 0.3`)
CLICK_FADE: float = 0.02        # s of gain fade at both ends against clicks


def spin_up_speed(progress: np.ndarray | float) -> np.ndarray:
    """The speed at a flash-in progress (0..1): 0 at the start, 1.5 at the peak, 1.16 at the end."""
    xs, ys = zip(*SPIN_UP)
    speed: np.ndarray = SPIN_UP_GAIN * np.interp(progress, xs, ys)
    return np.where(speed < SPIN_UP_FLOOR, 0.0, speed)


def speed_curve(duration: float, sample_rate: int, spin_up: float = 0.5, wind_down: float = 1.0) -> np.ndarray:
    """The speed of every output sample of a clip `duration` s long: the spin-up over the first `spin_up` s,
    1 in the middle, 1 to 0 over the last `wind_down` s."""
    n: int = max(1, round(duration * sample_rate))
    speed: np.ndarray = np.ones(n, np.float64)
    a: int = min(n, round(spin_up * sample_rate))
    if a > 0:
        speed[:a] = spin_up_speed(np.arange(a) / (spin_up * sample_rate))
    b: int = min(n, round(wind_down * sample_rate))
    if b > 0:
        speed[n - b:] = np.minimum(speed[n - b:], np.linspace(1.0, 0.0, b, endpoint=False))
    return speed


def resample(audio: np.ndarray, speed: np.ndarray, sample_rate: int) -> np.ndarray:
    """`audio` ((n, channels) float32) played at `speed` per output sample, as a tape does: the read position
    advances by the speed, pitch and tempo change together; past the end is silence. The gain fades in and
    out over `CLICK_FADE` s. Returns (len(speed), channels) float32."""
    position: np.ndarray = np.concatenate(([0.0], np.cumsum(speed)[:-1]))
    source: np.ndarray = np.arange(audio.shape[0], dtype=np.float64)
    out: np.ndarray = np.empty((len(speed), audio.shape[1]), np.float32)
    for channel in range(audio.shape[1]):
        out[:, channel] = np.interp(position, source, audio[:, channel].astype(np.float64), left=0.0, right=0.0)
    fade: int = min(len(speed) // 2, max(1, round(CLICK_FADE * sample_rate)))
    ramp: np.ndarray = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1, None]
    return out
