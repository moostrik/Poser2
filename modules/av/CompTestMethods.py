import numpy as np
import math
import time

def make_fill(resolution: int, white_strength: float, blue_strength: float) -> np.ndarray:
    return np.full((1, resolution, 3), [white_strength, blue_strength, 0.0], dtype=np.float16)

def make_pulse(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, pulse_speed: float) -> np.ndarray:
    def lfo(time_: float, Htz: float, phase: float) -> float:
        return 0.5 * math.sin(time_ * math.pi * Htz + phase) + 0.5

    T: float = time.time()
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    freq: float = pulse_speed if pulse_speed > 0 else 0.0  # pulses per second
    img[0, :, 0] = lfo(T, freq, 0.0) * white_strength
    img[0, :, 1] = lfo(T, freq, blue_phase * 2 * math.pi) * blue_strength
    return img

def make_chase(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, chase_speed: float, chase_amount: int) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    if resolution == 0 or chase_amount == 0:
        return img

    # adjusted_speed = chase_speed * math.pow(chase_amount, 0.33)
    adjusted_speed: float = chase_speed * math.log(chase_amount + 1)
    wave_phase_per_pixel: float = chase_amount * 2 * math.pi / resolution
    time_offset: float = time.time() * adjusted_speed * 2 * math.pi

    for i in range(resolution):
        WP: float = i * wave_phase_per_pixel - time_offset
        BP: float = i * wave_phase_per_pixel - time_offset + blue_phase * 2 * math.pi
        white_value: float = 0.5 * math.sin(WP) + 0.5
        blue_value: float = 0.5 * math.sin(BP) + 0.5
        img[0, i, 0] = white_value * white_strength
        img[0, i, 1] = blue_value * blue_strength
    return img

def make_lines(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, speed: float, amount: int, width: int) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    if resolution == 0 or amount == 0:
        return img

    adjusted_speed: float = speed / 10
    normalized_time = (time.time() * adjusted_speed) % 1.0

    for i in range(amount):
        white_pos: int = int(((normalized_time + i/amount) % 1.0) * resolution)
        blue_pos: int = int(((normalized_time + i/amount + blue_phase/amount) % 1.0) * resolution)

        for w in range(-width//2, width//2 + 1):
            pos: int = (white_pos + w) % resolution
            if 0 <= pos < resolution:
                img[0, pos, 0] = white_strength

        for w in range(-width//2, width//2 + 1):
            pos: int = (blue_pos + w) % resolution
            if 0 <= pos < resolution:
                img[0, pos, 1] = blue_strength

    return img

def make_random(resolution: int, white_strength: float, blue_strength: float, random_speed: float) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    if resolution == 0:
        return img
    T: float = time.time() * random_speed
    for i in range(resolution):
        if math.sin(T + i) > 0.5:
            img[0, i, 0] = white_strength
        if math.sin(T + i + 0.5) > 0.5:
            img[0, i, 1] = blue_strength
    return img
