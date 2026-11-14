"""The expansion's mean over the water's cells, and the divergence the solve then makes (numpy references of
expansion_mean_3d.comp and divergence_3d.comp's expansion)."""

import math

import numpy as np

GROUP: tuple[int, int, int] = (8, 8, 4)     # ExpansionMean3D's workgroup, x, y, z
LANES: int = 256                           # pass 1's one group


def expansion_mean(expansion: np.ndarray, obstacle: np.ndarray | None = None) -> float:
    """The mean of `expansion` (z, y, x) over the cells that are not an obstacle's (`obstacle` > 0.5), summed as
    the shader sums: per workgroup of 8 × 8 × 4 cells, then the partials 256 lanes at a time. Mirrors
    expansion_mean_3d.comp."""
    water: np.ndarray = np.ones_like(expansion, dtype=bool) if obstacle is None else obstacle <= 0.5
    depth, height, width = expansion.shape
    gx, gy, gz = GROUP
    partials: list[tuple[float, float]] = []
    for z in range(0, math.ceil(depth / gz) * gz, gz):
        for y in range(0, math.ceil(height / gy) * gy, gy):
            for x in range(0, math.ceil(width / gx) * gx, gx):
                cells = (slice(z, z + gz), slice(y, y + gy), slice(x, x + gx))
                partials.append((float(np.sum(expansion[cells] * water[cells])), float(np.sum(water[cells]))))
    lanes: np.ndarray = np.zeros((LANES, 2))
    for g, partial in enumerate(partials):
        lanes[g % LANES] += partial
    total, count = lanes.sum(axis=0)
    return float(total / count) if count > 0.0 else 0.0


def divergence_target(expansion: np.ndarray, rate: float, closed: bool, obstacle: np.ndarray | None = None) -> np.ndarray:
    """φ, the divergence the solve makes (1/s): `rate` times the expansion, in a closed box less its mean over the
    water's cells; 0 in an obstacle's cell. Mirrors divergence_3d.comp's expansion."""
    shifted: np.ndarray = expansion - (expansion_mean(expansion, obstacle) if closed else 0.0)
    phi: np.ndarray = rate * shifted
    return phi if obstacle is None else np.where(obstacle > 0.5, 0.0, phi)
