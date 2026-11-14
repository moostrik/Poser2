"""The peak on the GPU: the shader's speed bins against their numpy reference, the percentile `VelocityPeak` writes,
and the count's cost.

In a hidden window: a velocity of 448 × 512 speeds drawn on a log-normal spread around 0.5 widths/s, a fifth of
it still, so each sample hits a texel's center: the GPU's bins against `speed_counts` on the same speeds, which
must be equal. Then `VelocityPeak` fed that velocity for 1.5 s at 60 fps: the `speed` it writes against
`speed_percentile` of the same counts. Then the count's GPU time on deep_flow's 448 × 800 velocity, the median of
40 runs, and the copy's.

    python scripts/velocity_peak_check.py
"""

import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_NEAREST, GL_R32UI, GL_RED_INTEGER, GL_TEXTURE_2D, GL_UNSIGNED_INT, glBindTexture, glClearTexImage, glFinish, glGetTexImage  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.flow import VelocityPeak, VelocityPeakSettings, histogram_grid, speed_counts, speed_percentile  # noqa: E402
from modules.flow.visualization import HistogramCount  # noqa: E402
from modules.flow.bridge_3d.velocity_peak import BINS, SPEED_HIGH, SPEED_LOW  # noqa: E402
from modules.gl import Texture  # noqa: E402

FPS = 60.0


def velocity(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """A velocity (height, width, 4) and its speeds: directions at random, speeds log-normal about 0.5 widths/s,
    a fifth of the pixels still."""
    rng = np.random.default_rng(11)
    speeds = np.exp(rng.normal(np.log(0.5), 0.9, (height, width)))
    speeds[rng.random((height, width)) < 0.2] = 0.0
    direction = rng.normal(size=(height, width, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    values = np.zeros((height, width, 4), np.float32)
    values[..., :3] = direction * speeds[..., None]
    return values, np.linalg.norm(values[..., :3].astype(np.float64), axis=-1)


def gpu_bins(texture: Texture) -> np.ndarray:
    bins = Texture(GL_NEAREST)
    bins.allocate(BINS, 1, GL_R32UI)
    glClearTexImage(bins.tex_id, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, np.zeros(1, np.uint32))
    count = HistogramCount()
    count.allocate()
    count.use(texture, bins, 1.0, histogram_grid(texture.width, texture.height), (SPEED_LOW, SPEED_HIGH))
    glFinish()
    glBindTexture(GL_TEXTURE_2D, bins.tex_id)
    counts = np.asarray(glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT)).reshape(-1)[:BINS].astype(np.int64)
    glBindTexture(GL_TEXTURE_2D, 0)
    count.deallocate()
    bins.deallocate()
    return counts


def main() -> None:
    mh.context()
    values, speeds = velocity(448, 512)
    texture = mh.texture_of(values)
    try:
        gpu = gpu_bins(texture)
        reference = speed_counts(speeds.astype(np.float32).astype(np.float64))
        differ = int(np.abs(gpu - reference).sum())
        print(f'bins: {int(gpu.sum())} counted on the GPU, {int(reference.sum())} by numpy, {differ} counted in another bin', flush=True)

        settings = VelocityPeakSettings()
        settings.enabled = True
        peak = VelocityPeak(settings)
        peak.allocate()
        frames = 0
        for _ in range(int(1.5 * FPS)):
            peak.update(texture, 1.0 / FPS)
            frames += 1
            glFinish()
        expected = speed_percentile(reference)     # the same speeds every frame: a second's bins are the frame's times its count
        print(f'VelocityPeak: speed {settings.speed:.3f} widths/s, the counts\' percentile {expected:.3f}, '
              f'numpy\'s on the speeds {np.percentile(speeds[speeds >= SPEED_LOW], 99.0):.3f}', flush=True)
        settings.enabled = False
        peak.update(texture, 1.0 / FPS)
        print(f'off: speed {settings.speed}', flush=True)
        peak.deallocate()
    finally:
        texture.deallocate()

    values, _ = velocity(448, 800)
    texture = mh.texture_of(values)
    try:
        bins = Texture(GL_NEAREST)
        bins.allocate(BINS, 1, GL_R32UI)
        count = HistogramCount()
        count.allocate()
        grid = histogram_grid(texture.width, texture.height)
        ms = mh.gpu_ms(lambda: count.use(texture, bins, 1.0, grid, (SPEED_LOW, SPEED_HIGH)), 40)
        print(f'count on 448 x 800, {grid[0]} x {grid[1]} samples: {ms:.3f} ms', flush=True)
        settings = VelocityPeakSettings()
        settings.enabled = True
        peak = VelocityPeak(settings)
        peak.allocate()
        ms = mh.gpu_ms(lambda: peak._copy(0) or peak._release(), 40)
        print(f'copy into a pixel buffer: {ms:.3f} ms', flush=True)
        peak.deallocate()
        count.deallocate()
        bins.deallocate()
    finally:
        texture.deallocate()


if __name__ == '__main__':
    main()
