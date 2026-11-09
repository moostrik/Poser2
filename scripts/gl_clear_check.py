"""Does Texture.clear leave an RG32F FBO at exactly 0, every time, with memory churning around it?

In the app at startup a clear handed a temporary numpy array for its color sometimes left garbage (2026-11-09);
in this quiet harness it never did, so a pass here says the clear works, not that the app's clears do.

    python scripts/gl_clear_check.py
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_RG32F, glFinish  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.gl import SwapFbo  # noqa: E402

window = mh.context()
fbo = SwapFbo()
fbo.allocate(56, 100, GL_RG32F)
bad = 0
for i in range(3000):
    fbo.clear_all(7.0, -3.0, 0.0, 0.0)       # fill with a known value
    junk = [np.random.rand(1000) for _ in range(20)]     # churn memory like a running app
    del junk
    if i % 7 == 0:
        gc.collect()
    fbo.clear_all()
    glFinish()
    for side in (0, 1):
        values = fbo._fbos[side].read_to_numpy(flip=False)
        if values is None or not np.all(values[..., :2] == 0.0):
            bad += 1
            if bad <= 5:
                print('not zero after clear:', i, side, values[..., :2].reshape(-1, 2)[:2].tolist() if values is not None else None)
print(f'bad clears: {bad} of 6000')
