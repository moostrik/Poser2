"""Does the screen capture hand over the right NV12 frames from its mapped pool, and how fast is a frame copied out?

1. A 64 × 32 frame of 8 × 8 blocks of known colors is drawn into `ScreenCapture.frame`, captured, and handed over once
   its fence has passed: the Y and Cb Cr codes at each block's middle against the BT.709 video-range codes of its
   color (a table of numbers). `ScreenPreview` then draws the frame back through its thread and its mapped buffers:
   each block's middle holds its color within 3 levels of 255, the top row of blocks still at the top.
2. The pool: with no frame given back, four captures go out and the fifth is left out; after one is given back,
   the next frame says one was skipped before it.
3. The copy: one 2160 × 3840 frame copied out of its mapped buffer into plain RAM, as the recorder's thread does,
   timed over 20 copies; 60 frames a second need it well under 16.7 ms.

    python scripts/screen_capture_check.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.gl import Style  # noqa: E402
from modules.render import ScreenCapture, ScreenPreview  # noqa: E402
from modules.video import ScreenFrame  # noqa: E402

WIDTH, HEIGHT, BLOCK = 64, 32, 8

# color (0..255), and its BT.709 video-range Y, Cb, Cr codes
CODES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    'black':   ((0, 0, 0),       (16, 128, 128)),
    'white':   ((255, 255, 255), (235, 128, 128)),
    'red':     ((255, 0, 0),     (63, 102, 240)),
    'green':   ((0, 255, 0),     (173, 42, 26)),
    'blue':    ((0, 0, 255),     (32, 240, 118)),
    'yellow':  ((255, 255, 0),   (219, 16, 138)),
    'cyan':    ((0, 255, 255),   (188, 154, 16)),
    'magenta': ((255, 0, 255),   (78, 214, 230)),
}
TOP: list[str] = ['red', 'green', 'blue', 'white', 'black', 'yellow', 'cyan', 'magenta']
ROWS: list[list[str]] = [TOP] + [['black'] * 8] * (HEIGHT // BLOCK - 1)


def picture() -> np.ndarray:
    """The frame as GL holds it, row 0 at the bottom, RGBA float."""
    top_first = np.zeros((HEIGHT, WIDTH, 4), np.float32)
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            top_first[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK, :3] = np.array(CODES[name][0]) / 255.0
    top_first[..., 3] = 1.0
    return top_first[::-1].copy()


def draw(capture: ScreenCapture, source) -> None:
    Style.push_style()
    Style.set_blend_mode(Style.BlendMode.DISABLED)
    capture.frame.begin()
    source.draw()
    capture.frame.end()
    Style.pop_style()


def wait_for(capture: ScreenCapture, frames: list[ScreenFrame], count: int) -> None:
    glFinish()
    deadline = time.perf_counter() + 2.0
    while len(frames) < count and time.perf_counter() < deadline:
        capture.poll()
        glFinish()


def check_codes(source) -> int:
    bad = 0
    capture = ScreenCapture()
    capture.allocate(WIDTH, HEIGHT)
    frames: list[ScreenFrame] = []
    capture.add_screen_frame_callback(frames.append)
    draw(capture, source)
    capture.capture(1.0)
    wait_for(capture, frames, 1)
    if not frames:
        print('  no frame handed over')
        return 1
    nv12 = frames[0].data.copy()
    frames[0].release()
    luma, chroma = nv12[:HEIGHT], nv12[HEIGHT:].reshape(HEIGHT // 2, WIDTH // 2, 2)
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            y, x = r * BLOCK + BLOCK // 2, c * BLOCK + BLOCK // 2
            got = (int(luma[y, x]), int(chroma[y // 2, x // 2, 0]), int(chroma[y // 2, x // 2, 1]))
            want = CODES[name][1]
            if any(abs(g - w) > 1 for g, w in zip(got, want)):
                bad += 1
                print(f'  NV12 {name} at row {r} col {c}: {got}, want {want}')
    preview = ScreenPreview()
    preview.allocate(WIDTH, HEIGHT)
    preview.set_screen_frame(0, nv12)
    deadline = time.perf_counter() + 2.0
    while preview.shown != 0 and time.perf_counter() < deadline:     # its thread copies, the next update draws
        preview.update()
        time.sleep(0.001)
    if preview.shown != 0:
        print('  the preview drew nothing')
        return bad + 1
    glFinish()
    shown = preview.fbo.read_to_numpy(flip=False)
    preview.deallocate()
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
    capture.deallocate()
    return bad


def check_pool(source) -> int:
    bad = 0
    capture = ScreenCapture()
    capture.allocate(WIDTH, HEIGHT)
    frames: list[ScreenFrame] = []
    capture.add_screen_frame_callback(frames.append)
    for n in range(5):
        draw(capture, source)
        capture.capture(float(n))
    wait_for(capture, frames, 4)
    if len(frames) != 4:
        bad += 1
        print(f'  {len(frames)} frames out of five captures with four buffers, want 4')
    frames[0].release()
    draw(capture, source)
    capture.capture(5.0)
    wait_for(capture, frames, 5)
    if len(frames) != 5 or frames[-1].skipped != 1:
        bad += 1
        print(f'  after one given back: {len(frames)} frames, skipped {frames[-1].skipped if frames else None}, want 5 and 1')
    if not capture.lent:
        bad += 1
        print('  the pool says nothing is lent while four frames are')
    for frame in frames[1:]:
        frame.release()
    if capture.lent:
        bad += 1
        print('  the pool says a frame is lent after all were given back')
    capture.deallocate()
    return bad


def time_copy() -> float:
    capture = ScreenCapture()
    capture.allocate(2160, 3840)
    frames: list[ScreenFrame] = []
    capture.add_screen_frame_callback(frames.append)
    capture.capture(0.0)
    wait_for(capture, frames, 1)
    ring = np.empty_like(frames[0].data)
    np.copyto(ring, frames[0].data)         # first touch of the destination's pages, as the ring's new block has
    started = time.perf_counter()
    for _ in range(20):
        np.copyto(ring, frames[0].data)
    ms = (time.perf_counter() - started) / 20 * 1000.0
    frames[0].release()
    capture.deallocate()
    return ms


window = mh.context()
source = mh.texture_of(picture())
codes = check_codes(source)
print(f'NV12 codes and preview: {codes} wrong of {2 * len(ROWS) * len(TOP)}')
pool = check_pool(source)
print(f'pool: {pool} wrong')
ms = time_copy()
mb = 2160 * 3840 * 3 // 2 / 1e6
print(f'copy of one 2160x3840 NV12 frame ({mb:.1f} MB) out of a mapped buffer: {ms:.2f} ms ({mb / ms:.1f} GB/s)')
print('PASS' if codes == 0 and pool == 0 and ms < 8.0 else 'FAIL')
