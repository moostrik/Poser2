"""Does a screen frame reach the ring as the right NV12, and come back from it upright as the same RGB, in RAM and in
video memory?

A 64 × 32 frame of 8 × 8 blocks of known colors is drawn into `ScreenCapture.frame`, captured into slot 0 of a ring
of two, and read back: the Y and Cb Cr codes at each block's middle against the BT.709 video-range codes of its color
(a table of numbers, not a second conversion). Then `show_preview(0)` draws the slot back into `preview`, and each
block's middle must hold its color within 3 levels of 255, the top row of blocks still at the top.

    python scripts/screen_capture_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.gl import ScreenCapture, Style  # noqa: E402

WIDTH, HEIGHT, BLOCK = 64, 32, 8

# color (0..255), and its BT.709 video-range Y, Cb, Cr codes
CODES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    'black':  ((0, 0, 0),       (16, 128, 128)),
    'white':  ((255, 255, 255), (235, 128, 128)),
    'red':    ((255, 0, 0),     (63, 102, 240)),
    'green':  ((0, 255, 0),     (173, 42, 26)),
    'blue':   ((0, 0, 255),     (32, 240, 118)),
    'yellow': ((255, 255, 0),   (219, 16, 138)),
    'cyan':   ((0, 255, 255),   (188, 154, 16)),
    'magenta': ((255, 0, 255),  (78, 214, 230)),
}
# the picture's rows of blocks, top first: the top row differs from the others, so a flip shows
TOP: list[str] = ['red', 'green', 'blue', 'white', 'black', 'yellow', 'cyan', 'magenta']
BELOW: list[str] = ['black'] * 8
ROWS: list[list[str]] = [TOP] + [BELOW] * (HEIGHT // BLOCK - 1)


def picture() -> np.ndarray:
    """The frame as GL holds it, row 0 at the bottom, RGBA float."""
    top_first = np.zeros((HEIGHT, WIDTH, 4), np.float32)
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            top_first[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK, :3] = np.array(CODES[name][0]) / 255.0
    top_first[..., 3] = 1.0
    return top_first[::-1].copy()


def check(host_memory: bool, source) -> int:
    bad = 0
    capture = ScreenCapture()
    capture.allocate(WIDTH, HEIGHT, host_memory)
    capture.add_block(2)
    Style.push_style()
    Style.set_blend_mode(Style.BlendMode.DISABLED)
    capture.frame.begin()
    source.draw()
    capture.frame.end()
    Style.pop_style()
    capture.capture(0)
    capture.flush()
    capture.synchronize()
    capture.queue_download(0)
    nv12 = capture.wait_download().copy()
    luma, chroma = nv12[:HEIGHT], nv12[HEIGHT:].reshape(HEIGHT // 2, WIDTH // 2, 2)
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            y, x = r * BLOCK + BLOCK // 2, c * BLOCK + BLOCK // 2
            got = (int(luma[y, x]), int(chroma[y // 2, x // 2, 0]), int(chroma[y // 2, x // 2, 1]))
            want = CODES[name][1]
            if any(abs(g - w) > 1 for g, w in zip(got, want)):
                bad += 1
                print(f'  NV12 {name} at row {r} col {c}: {got}, want {want}')
    capture.show_preview(0)
    glFinish()
    shown = capture.preview.read_to_numpy(flip=False)
    assert shown is not None
    top_first = shown[::-1]
    scale = 255.0 if shown.dtype != np.uint8 else 1.0
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            y, x = r * BLOCK + BLOCK // 2, c * BLOCK + BLOCK // 2
            got = tuple(int(round(float(v) * scale)) for v in top_first[y, x, :3])
            want = CODES[name][0]
            if any(abs(g - w) > 3 for g, w in zip(got, want)):
                bad += 1
                print(f'  preview {name} at row {r} col {c}: {got}, want {want}')
    capture.free_blocks()
    capture.deallocate()
    return bad


window = mh.context()
source = mh.texture_of(picture())
total = 0
for host_memory in (True, False):
    bad = check(host_memory, source)
    total += bad
    print(f'{"RAM" if host_memory else "video memory"}: {bad} wrong of {2 * len(ROWS) * len(TOP)}')
print('PASS' if total == 0 else f'FAIL: {total} wrong')
