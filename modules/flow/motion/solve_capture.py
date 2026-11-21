"""SolveCapture: keeps the Lucas–Kanade solve's inputs when its output comes out wrong, for a replay outside the app.

Off unless the environment variable POSER_SOLVE_CAPTURE names a folder. Then before each solve it reads the two
luminance frames at every mip level the solve uses and the depth (a readback that waits for the GPU, so a
debugging tool only), and after it, where the solve holds a non-finite value or a step larger than half the
image a frame, it saves them with the solve and its settings as `solve_capture_<n>.npz`, at most
`MAX_CAPTURES` a run. `scripts/solve_replay.py` reruns a capture through the GPU solve.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from modules.gl import Texture

logger = logging.getLogger(__name__)

ENVIRONMENT: str = 'POSER_SOLVE_CAPTURE'
MAX_CAPTURES: int = 3
WRONG_STEP: float = 0.5         # uv a frame: a solve step past half the image is no motion


def read_level(texture: Texture, lod: int) -> np.ndarray:
    """One mip level of a texture's first channel, in its own type (rows bottom first, as GL holds them)."""
    pixels = texture.read_to_numpy(flip=False, level=lod)
    if pixels is None:
        return np.zeros((0, 0), np.float32)
    return pixels[..., 0] if pixels.ndim == 3 else pixels


class SolveCapture:
    """Captures the solve's inputs before it runs and keeps them when its output is wrong."""

    def __init__(self, folder: Path) -> None:
        self._folder: Path = folder
        self._saved: int = 0

    @staticmethod
    def from_environment() -> SolveCapture | None:
        folder: str | None = os.environ.get(ENVIRONMENT)
        if not folder:
            return None
        Path(folder).mkdir(parents=True, exist_ok=True)
        logger.warning('Solve capture on: bad solves are saved to %s', folder)
        return SolveCapture(Path(folder))

    def before(self, previous: Texture, current: Texture, levels: int, depth: Texture | None) -> dict[str, np.ndarray] | None:
        """The solve's inputs as they are now: both luminance frames at each of `levels` mip levels, and the depth."""
        if self._saved >= MAX_CAPTURES:
            return None
        inputs: dict[str, np.ndarray] = {}
        for lod in range(levels):
            inputs[f'current_{lod}'] = read_level(current, lod)
            inputs[f'previous_{lod}'] = read_level(previous, lod)
        if depth is not None and depth.allocated:
            inputs['depth'] = read_level(depth, 0)
        return inputs

    def after(self, inputs: dict[str, np.ndarray] | None, solved: Texture, settings: dict[str, float]) -> None:
        """Save `inputs` with the solve and `settings` when the solve holds a non-finite value or an impossible step."""
        if inputs is None:
            return
        values = solved.read_to_numpy(flip=False)
        if values is None:
            return
        uv = values[..., :2]
        wrong = int(np.count_nonzero(~np.isfinite(uv))) + int(np.count_nonzero(np.abs(np.nan_to_num(uv)) > WRONG_STEP))
        if wrong == 0:
            return
        self._saved += 1
        path: Path = self._folder / f'solve_capture_{self._saved}.npz'
        np.savez_compressed(path, allow_pickle=True, solved=values, **settings, **inputs)
        logger.warning('Solve capture: %d wrong values, saved to %s', wrong, path)
