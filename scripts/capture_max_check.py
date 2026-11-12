"""Does `CaptureMax` keep, per pixel, the output of the capture with the larger amount?

Three captures' densities (RGBA, the amount in alpha) and temperatures (±, the amount its size) with known values
per pixel; the result is compared with numpy's pick. Needs a GL context (a hidden window).

    python scripts/capture_max_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_R16F, GL_RED, GL_RGBA, GL_RGBA16F, GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage, glTexSubImage2D  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.flow import CaptureMax  # noqa: E402
from modules.gl import Texture  # noqa: E402

SIZE = (16, 8)


def texture(values: np.ndarray, internal_format: int) -> Texture:
    t = Texture()
    t.allocate(SIZE[0], SIZE[1], internal_format)
    glBindTexture(GL_TEXTURE_2D, t.tex_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIZE[0], SIZE[1], GL_RGBA if values.ndim == 3 else GL_RED, GL_FLOAT, np.ascontiguousarray(values, np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return t


def read(t: Texture, channels: int) -> np.ndarray:
    glFinish()
    glBindTexture(GL_TEXTURE_2D, t.tex_id)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA if channels == 4 else GL_RED, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    return np.frombuffer(data, np.float32).reshape((SIZE[1], SIZE[0], channels) if channels == 4 else (SIZE[1], SIZE[0]))


def main() -> int:
    window = mh.context()
    rng = np.random.default_rng(3)
    densities = [rng.random((SIZE[1], SIZE[0], 4)).astype(np.float32) for _ in range(3)]
    temperatures = [rng.uniform(-1.0, 1.0, (SIZE[1], SIZE[0])).astype(np.float32) for _ in range(3)]
    failed = 0
    for name, values, by_alpha, internal_format, channels in (('density', densities, True, GL_RGBA16F, 4),
                                                              ('temperature', temperatures, False, GL_R16F, 1)):
        capture_max = CaptureMax(by_alpha)
        textures = [texture(v, internal_format) for v in values]
        uploaded = [read(t, channels) for t in textures]             # as 16-bit floats hold them
        capture_max.hold(textures[0])
        for t in textures[1:]:
            capture_max.take(t)
        got = read(capture_max.texture, channels)
        amounts = np.stack([u[..., 3] if by_alpha else np.abs(u) for u in uploaded])
        pick = np.argmax(amounts, axis=0)
        stacked = np.stack(uploaded)
        want = np.take_along_axis(stacked, pick[None, ..., None].repeat(4, axis=-1), axis=0)[0] if by_alpha else \
            np.take_along_axis(stacked, pick[None], axis=0)[0]
        off = np.abs(got - want) > 1e-3
        wrong = int((off.any(axis=-1) if by_alpha else off).sum())
        print(f'{name}: {wrong} of {pick.size} pixels not the larger capture\'s')
        failed += wrong
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
