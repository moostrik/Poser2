"""Does `gl.Image` with a `src` draw as it must: the identity `src` as the plain upload, and a camera frame's
`oriented_corners` as the plain upload of its `oriented_image`?

An odd-sized BGR image (rows of 21 bytes, not a multiple of 4) through every flag set; each texture is read back and
compared exactly.

    python scripts/gl_image_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.camera import CameraFrame, FrameType, oriented_corners, oriented_image  # noqa: E402
from modules.gl import ChannelOrder, Image  # noqa: E402


def uploaded(image: np.ndarray, frame: CameraFrame | None = None) -> np.ndarray:
    """The image's texture as read back: plain, or with the frame's `oriented_corners` as its `src`."""
    texture = Image(ChannelOrder.BGR)
    texture.set_from_image(image, oriented_corners(frame) if frame is not None else None)
    glFinish()
    pixels = texture.read_to_numpy()
    texture.deallocate()
    assert pixels is not None
    return pixels


window = mh.context()
image = np.random.default_rng(0).integers(0, 255, (5, 7, 3), np.uint8)
failed = 0
for rotate90 in (False, True):
    for flip_v in (False, True):
        for flip_h in (False, True):
            frame = CameraFrame(0, FrameType.VIDEO, image, 0.0, rotate90=rotate90, flip_v=flip_v, flip_h=flip_h)
            expected = uploaded(np.ascontiguousarray(oriented_image(frame)))
            drawn = uploaded(image, frame)
            same = drawn.shape == expected.shape and np.array_equal(drawn, expected)
            failed += not same
            print(f'rotate90 {rotate90!s:5} flip_v {flip_v!s:5} flip_h {flip_h!s:5}: {drawn.shape} '
                  f'{"same" if same else "DIFFERENT"} (largest {int(np.abs(drawn.astype(int) - expected.astype(int)).max()) if drawn.shape == expected.shape else "-"})')
print('all the same' if not failed else f'{failed} different')
sys.exit(1 if failed else 0)
