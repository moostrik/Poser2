import numpy as np
import math
import time

def make_fill(resolution: int, strength: float) -> np.ndarray:
    return np.full((1, resolution), strength, dtype=np.float16)

def make_pulse(resolution: int, strength: float, speed: float, phase: float) -> np.ndarray:
    def lfo(time_: float, Htz: float, phase: float) -> float:
        return 0.5 * math.sin(time_ * math.pi * Htz + phase) + 0.5

    T: float = time.time()
    img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
    freq: float = speed
    img[0, :] = lfo(T, freq, phase * 2 * math.pi) * strength
    return img

def make_chase(resolution: int, strength: float, speed: float, phase: float, amount: int) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
    if resolution == 0 or amount == 0:
        return img

    # adjusted_speed = chase_speed * math.pow(chase_amount, 0.33)
    # adjusted_speed: float = speed * math.log(amount + 1)
    adjusted_speed: float = speed * amount / 10.0
    wave_phase_per_pixel: float = amount * 2 * math.pi / resolution
    time_offset: float = time.time() * adjusted_speed * 2 * math.pi

    for i in range(resolution):
        P: float = i * wave_phase_per_pixel - time_offset + phase * 2 * math.pi
        value: float = 0.5 * math.sin(P) + 0.5
        img[0, i] = value * strength
    return img

def make_lines(resolution: int, strength: float, speed: float, phase: float, amount: int, width: float) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
    if resolution == 0 or amount == 0:
        return img

    adjusted_speed: float = speed * amount / 10.0
    wave_phase_per_pixel: float = amount * 2 * math.pi / resolution
    time_offset: float = time.time() * adjusted_speed * 2 * math.pi

    for i in range(resolution):
        P: float = i * wave_phase_per_pixel - time_offset + phase * 2 * math.pi + math.pi
        value: float = 0.5 * math.sin(P) + 0.5
        value = 1.0 if value < width else 0.0
        img[0, i] = value * strength
    return img

def make_random(resolution: int, strength: float, speed: float) -> np.ndarray:
    img: np.ndarray = np.zeros((1, resolution), dtype=np.float16)
    if resolution == 0:
        return img
    T: float = time.time() * speed
    for i in range(resolution):
        if math.sin(T + i) > 0.5:
            img[0, i] = strength
    return img
