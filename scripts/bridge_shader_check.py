"""The 3D bridge's GL shaders against their numpy references, on random fields: the largest difference of each.

No app and no camera: a hidden window gives the GL context.

    python scripts/bridge_shader_check.py
"""

import sys
from pathlib import Path

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_RG, GL_RG32F, GL_RGBA, GL_RGBA16F, GL_TEXTURE_2D, glBindTexture, glTexImage2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Texture  # noqa: E402
from modules.flow.bridge_3d.shaders import LowpassBlur  # noqa: E402
from modules.flow.bridge_3d.velocity_lowpass import blur_dispatches, gaussian_blur, gaussian_weights  # noqa: E402
from depth_velocity_compare import context, read_back  # noqa: E402

HALF_FLOAT: float = 2e-3    # the relative step of RGBA16F near 1, and some


def texture(array: np.ndarray, internal_format: int) -> Texture:
    """A float texture of `array` (H, W, C), C 2 or 4."""
    h, w, channels = array.shape
    result = Texture()
    result.allocate(w, h, internal_format)
    glBindTexture(GL_TEXTURE_2D, result.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RG if channels == 2 else GL_RGBA, GL_FLOAT,
                 np.ascontiguousarray(array, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return result


def check_lowpass_blur(rng: np.random.Generator) -> bool:
    shader = LowpassBlur()
    shader.allocate()
    passed = True
    h, w = 61, 45      # not a multiple of the tile
    for sigma in (0.4, 1.0, 2.0, 2.6, 5.0):
        fields = [rng.uniform(-1.0, 1.0, (h, w, 4)), rng.uniform(-1.0, 1.0, (h, w, 4)), rng.uniform(0.0, 1.0, (h, w, 2))]
        formats = [GL_RGBA16F, GL_RGBA16F, GL_RG32F]
        sources = [texture(f, fmt) for f, fmt in zip(fields, formats)]
        targets = [texture(np.zeros_like(f), fmt) for f, fmt in zip(fields, formats)]
        count, each = blur_dispatches(sigma, LowpassBlur.MAX_RADIUS)
        weights = gaussian_weights(each).tolist()
        expected = [f.copy() for f in fields]
        for step in range(count):
            shader.use(sources if step % 2 == 0 else targets, targets if step % 2 == 0 else sources, weights)
            expected = [gaussian_blur(e, each) for e in expected]
        result = targets if count % 2 == 1 else sources
        for name, out, exp in zip(('first', 'second', 'third'), result, expected):
            got = read_back(out)[..., :exp.shape[2]]
            diff = float(np.abs(got - exp).max())
            ok = diff < HALF_FLOAT
            passed &= ok
            print(f"lowpass blur, sigma {sigma:.1f} px in {count} dispatch(es), {name} field: max diff {diff:.2e} {'ok' if ok else 'FAIL'}")
    return passed


def main() -> None:
    context(False, 64, 64)
    rng = np.random.default_rng(7)
    passed = check_lowpass_blur(rng)
    print('all ok' if passed else 'FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
