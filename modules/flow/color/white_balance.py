"""The white balance: the 3 × 3 matrix, on linear sRGB, that adapts the white of a blackbody light at a temperature
(kelvin) to a 6500 K blackbody's by Bradford's chromatic adaptation; the identity at 6500 K."""

import numpy as np

REFERENCE_KELVIN: float = 6500.0

# Linear sRGB (D65) to CIE XYZ, and back
RGB_TO_XYZ: np.ndarray = np.array([[0.4124564, 0.3575761, 0.1804375],
                                   [0.2126729, 0.7151522, 0.0721750],
                                   [0.0193339, 0.1191920, 0.9503041]])
_XYZ_TO_RGB: np.ndarray = np.linalg.inv(RGB_TO_XYZ)

# XYZ to Bradford's cone response
_BRADFORD: np.ndarray = np.array([[0.8951, 0.2664, -0.1614],
                                  [-0.7502, 1.7135, 0.0367],
                                  [0.0389, -0.0685, 1.0296]])
_BRADFORD_INVERSE: np.ndarray = np.linalg.inv(_BRADFORD)


def planckian_white(kelvin: float) -> np.ndarray:
    """The white of a blackbody at `kelvin` (1667 to 25000 K), XYZ with Y = 1: its chromaticity on the Planckian
    locus by Kim et al.'s cubic spline (2002)."""
    t: float = float(np.clip(kelvin, 1667.0, 25000.0))
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def white_balance_matrix(kelvin: float) -> np.ndarray:
    """The matrix on linear sRGB that turns the white of a light at `kelvin` into the 6500 K light's, its
    brightness (Y) kept: lower cools the image, higher warms it."""
    source: np.ndarray = _BRADFORD @ planckian_white(kelvin)
    target: np.ndarray = _BRADFORD @ planckian_white(REFERENCE_KELVIN)
    adapt: np.ndarray = _BRADFORD_INVERSE @ np.diag(target / source) @ _BRADFORD
    return (_XYZ_TO_RGB @ adapt @ RGB_TO_XYZ).astype(np.float32)
