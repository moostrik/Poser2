"""The 3D fluid's injections at each pixel's z position against the slab they should fill: the largest difference.

A 2D source of ones goes into a volume by `Inject3D` and `InjectAmount3D` (ADD, strength 1), each column at its
z position, half the columns without one (nothing goes in there); each cell should hold the body's share of it,
the gaussian's area between its faces over its length (slab.glsl). Without a z position bound nothing goes in.
No app: a hidden window gives the GL context.

    python scripts/injection_check.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_R16F, GL_RG, GL_RG32F, GL_RGBA, GL_RGBA16F, GL_TEXTURE_2D, glBindTexture, glFinish, glTexImage2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture, Texture3D  # noqa: E402
from modules.flow.fluid_3d.shaders import Inject3D, InjectAmount3D  # noqa: E402
from depth_velocity_compare import context  # noqa: E402


def texture(array: np.ndarray, internal_format: int) -> Texture:
    h, w, channels = array.shape
    result = Texture()
    result.allocate(w, h, internal_format)
    glBindTexture(GL_TEXTURE_2D, result.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RG if channels == 2 else GL_RGBA, GL_FLOAT,
                 np.ascontiguousarray(array, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return result


def slab_cells(center: float, spread: float, layers: int) -> np.ndarray:
    """Each layer's share of a body at `center` with sigma `spread` (shares of the depth), as slab.glsl gives it:
    the profile (1 at its middle) averaged over the cell."""
    middle: float = (0.5 + center * (layers - 1)) / layers
    faces = np.arange(layers + 1) / layers
    cdf = np.array([0.5 * (1.0 + math.erf((f - middle) / (spread * math.sqrt(2.0)))) for f in faces])
    return spread * math.sqrt(2.0 * math.pi) * np.diff(cdf) * layers       # the area over a cell of 1 / layers


def main() -> None:
    context(False, 64, 64)
    w, h, layers, spread = 24, 16, 24, 0.04
    rng = np.random.default_rng(3)
    places = rng.uniform(0.0, 1.0, (h, w))
    weights = np.where(np.arange(w)[None, :] % 2 == 0, rng.uniform(0.2, 1.0, (h, w)), 0.0)    # odd columns: none
    z_position = texture(np.stack((places * weights, weights), -1), GL_RG32F)
    source = texture(np.ones((h, w, 4)), GL_RGBA16F)
    expected = np.stack([np.stack([slab_cells(places[y, x], spread, layers) * (weights[y, x] > 0.0) for x in range(w)])
                         for y in range(h)])   # (h, w, layers); a column without a z position holds nothing
    expected = np.where(expected * 1.0 < 0.001, 0.0, expected)      # a share below this is skipped

    passed = True
    inject = Inject3D()
    inject.allocate()
    volume = Texture3D()
    volume.allocate(w, h, layers, GL_RGBA16F)
    volume.clear()
    inject.use(source, volume, spread, 1.0, internal_format=GL_RGBA16F, mode=0, z_position=z_position)
    glFinish()
    got = volume.read_to_numpy()
    assert got is not None
    diff = float(np.abs(np.moveaxis(got[..., 3], 0, -1) - expected).max())
    passed &= diff < 5e-3
    print(f"inject, density at each column's z position: max diff {diff:.2e} {'ok' if diff < 5e-3 else 'FAIL'}")

    amount = InjectAmount3D()
    amount.allocate()
    finer = Texture3D()
    finer.allocate(w, h, layers, GL_R16F)
    finer.clear()
    amount.use(source, finer, spread, 1.0, z_position=z_position)
    glFinish()
    got = finer.read_to_numpy()
    assert got is not None
    diff = float(np.abs(np.moveaxis(got, 0, -1) - expected).max())
    passed &= diff < 5e-3
    print(f"inject amount at each column's z position: max diff {diff:.2e} {'ok' if diff < 5e-3 else 'FAIL'}")

    volume.clear()
    inject.use(source, volume, spread, 1.0, internal_format=GL_RGBA16F, mode=0)
    glFinish()
    got = volume.read_to_numpy()
    assert got is not None
    diff = float(np.abs(got[..., 3]).max())
    passed &= diff < 5e-3
    print(f"inject without a z position, nothing in: max {diff:.2e} {'ok' if diff < 5e-3 else 'FAIL'}")
    print('all ok' if passed else 'FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
