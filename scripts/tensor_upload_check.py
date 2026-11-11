"""Does a CUDA tensor reach its GL texture exactly through `Tensor`, the copy queued on the stream and not waited for?

Each format the render takes in (an RGB float16 image, an R32F depth, an RGBA float16 density) gets a new random tensor
every round, written on a side stream as the uploader writes its images, waited for by the render's stream through an
event, taken into GL, and read back. A round with GPU work queued in front of it on the side stream checks that the
copy waits for the tensor it copies.

    python scripts/tensor_upload_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from OpenGL.GL import GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage, GL_RGBA, GL_FLOAT  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.gl import Tensor  # noqa: E402

window = mh.context()
side = torch.cuda.Stream()
cases = {
    'image (3, 240, 320) float16': (3, 240, 320, torch.float16),
    'depth (240, 320) float32': (None, 240, 320, torch.float32),
    'density (4, 120, 160) float16': (4, 120, 160, torch.float16),
}
bad = 0
for name, (channels, height, width, dtype) in cases.items():
    texture = Tensor()
    for round_ in range(40):
        with torch.cuda.stream(side):
            if round_ % 4 == 0:     # GPU work in front of the tensor, so the copy has to wait for it
                busy = torch.randn(2048, 2048, device='cuda')
                for _ in range(20):
                    busy = busy @ busy
                    busy = busy / busy.norm()
            shape = (height, width) if channels is None else (channels, height, width)
            tensor = torch.rand(shape, device='cuda').to(dtype)
            ready = torch.cuda.Event()
            ready.record(side)
        render = torch.cuda.current_stream()
        render.wait_event(ready)
        tensor.record_stream(render)
        texture.set_tensor(tensor)
        texture.update()
        glFinish()
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)
        read = np.frombuffer(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT), np.float32).reshape(height, width, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        read = read[::-1]                                           # GL rows run up, the tensor's down
        expected = tensor.float().cpu().numpy()
        expected = expected[..., None] if channels is None else np.moveaxis(expected, 0, -1)
        got = read[..., :expected.shape[-1]]
        if not np.array_equal(got, expected):
            bad += 1
            if bad <= 5:
                print(f'{name}, round {round_}: largest difference {np.abs(got - expected).max()}')
    texture.deallocate()
print(f'bad rounds: {bad} of {40 * len(cases)}')
sys.exit(1 if bad else 0)
