import numpy as np
import math
import time

def make_fill(resolution: int, white_strength: float, blue_strength: float) -> np.ndarray:
    return np.full((1, resolution, 3), [white_strength, blue_strength, 0.0], dtype=np.float16)

def make_pulse(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, pulse_speed: float) -> np.ndarray:
    T: float = time.time() * pulse_speed
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    img[0, :, 0] = lfo(T, 1.0 / 2.0, 0.0) * white_strength
    img[0, :, 1] = lfo(T, 1.0 / 2.0, blue_phase) * blue_strength
    return img

def make_chase(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, chase_speed: float, chase_amount: int) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    if resolution == 0:
        return img
    T: float = time.time()
    W_P: float = 1.0 / resolution * chase_amount
    B_P: float = 1.0 / resolution * chase_amount + blue_phase
    for i in range(resolution):
        white_value: float = lfo(T, chase_speed / (math.pi * 2.0), i * W_P)
        blue_value: float = lfo(T, chase_speed / (math.pi * 2.0), i * B_P)
        img[0, i, 0] = white_value * white_strength
        img[0, i, 1] = blue_value * blue_strength
    return img

def make_single(resolution: int, white_strength: float, blue_strength: float, blue_phase: float, single_speed: float, single_amount: int) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
    if resolution == 0 or single_amount == 0:
        return img
    interval: int = max(1, int(resolution / single_amount))
    t_idx: int = int(time.time() * single_speed)
    for i in range(resolution):
        if t_idx % interval == i % interval:
            img[0, i, 0] = white_strength
        if ((t_idx + interval // 2) % interval) == (i % interval):
            img[0, i, 1] = blue_strength
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

def lfo(time_: float, Htz: float, phase: float) -> float:
    return 0.5 * math.sin(time_ * math.pi * Htz + (0.75 + phase) * math.pi * 2) + 0.5

def lfor(time_: float, Htz: float, phase: float, rangeMin: float, rangeMax: float) -> float:
    return lfo(time_, Htz, phase) * (rangeMax - rangeMin) + rangeMin