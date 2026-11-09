"""Replay a captured Lucas–Kanade solve: its inputs checked, then the GPU solve and the numpy reference on them.

Capture first: run the app with POSER_SOLVE_CAPTURE set to a folder (`modules/flow/motion/solve_capture.py`); a
solve that comes out wrong is saved there as solve_capture_<n>.npz with its inputs and settings. Then:

    python scripts/solve_replay.py <folder or .npz files> [--runs 50]

For each capture it prints what the app's solve gave, whether each luminance mip level matches its full image
downscaled, the depth's range, the numpy reference with the same settings, and how many of `--runs` GPU solves
in a hidden window come out wrong. A capture the GPU solves right here but wrong in the app points at the app's
state at that moment, not at the solve.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import GL_FLOAT, GL_R32F, GL_RED, GL_TEXTURE_2D, glBindTexture, glFinish, glTexImage2D  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.flow import LucasKanadeFlow, LucasKanadeFlowSettings, Patch, lucas_kanade  # noqa: E402
from modules.gl import Texture  # noqa: E402

WRONG_STEP: float = 0.5     # uv a frame, as the capture judges


def image_texture(gray: np.ndarray) -> Texture:
    rgba = np.zeros(gray.shape + (4,), np.float32)
    rgba[..., :3] = gray[..., None].astype(np.float32) / 255.0
    rgba[..., 3] = 1.0
    return mh.texture_of(rgba)


def depth_texture(depth: np.ndarray) -> Texture:
    texture = Texture()
    h, w = depth.shape
    texture.allocate(w, h, GL_R32F)
    glBindTexture(GL_TEXTURE_2D, texture.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, np.ascontiguousarray(depth, np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def wrong_count(uv: np.ndarray) -> int:
    return int((~np.isfinite(uv)).sum() + (np.abs(np.nan_to_num(uv)) > WRONG_STEP).sum())


def report(path: Path, runs: int) -> None:
    f = np.load(path)
    levels, window, damping = int(f['levels']), int(f['window']), float(f['damping'])
    same_surface, tolerance = bool(f['same_surface']), float(f['tolerance'])
    print(f'--- {path.name}: levels {levels}, window {window}, damping {damping}, same_surface {same_surface}')
    solved = f['solved'][..., :2]
    print(f'  the app\'s solve: {wrong_count(solved)} wrong values of {solved.size}')
    for name in ('previous', 'current'):
        base = f[f'{name}_0'].astype(np.float32) / 255
        for lod in range(1, levels):
            image = f[f'{name}_{lod}'].astype(np.float32) / 255
            h, w = image.shape
            k = 2 ** lod
            expected = base[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))
            print(f'  {name} mip {lod} {image.shape}: largest gap to the full image downscaled {np.abs(image - expected).max():.3f}')
    flow_depth = None
    if 'depth' in f.files:
        full = f['depth']
        print(f'  depth {full.shape}: non-finite {int((~np.isfinite(full)).sum())}, range {float(np.nanmin(full)):.2f} to {float(np.nanmax(full)):.2f} m')
        fh, fw = f['current_0'].shape
        rows = np.minimum(((np.arange(fh) + 0.5) / fh * full.shape[0]).astype(int), full.shape[0] - 1)
        cols = np.minimum(((np.arange(fw) + 0.5) / fw * full.shape[1]).astype(int), full.shape[1] - 1)
        flow_depth = full[rows][:, cols].astype(np.float64)
    reference = lucas_kanade(f['previous_0'].astype(np.float64) / 255, f['current_0'].astype(np.float64) / 255, levels=levels,
                             window=window, damping=damping, depth=flow_depth if same_surface else None, tolerance=tolerance)
    print(f'  numpy reference: non-finite {int((~np.isfinite(reference)).sum())}, largest step {float(np.nanmax(np.abs(reference))):.3f} px')

    settings = LucasKanadeFlowSettings()
    settings.levels, settings.window, settings.damping, settings.same_surface = levels, Patch(window), damping, same_surface
    flow = LucasKanadeFlow(settings)
    h, w = f['current_0'].shape
    flow.allocate(w, h)
    previous, current = image_texture(f['previous_0']), image_texture(f['current_0'])
    depth = depth_texture(f['depth']) if 'depth' in f.files else None
    wrong_runs = 0
    for _ in range(runs):
        flow.reset()
        flow.set_input(previous)
        flow.set_input(current)
        flow.update(depth, tolerance)
        glFinish()
        values = flow.solved.read_to_numpy(flip=False)
        if values is not None and wrong_count(values[..., :2]):
            wrong_runs += 1
    print(f'  GPU solve here: {wrong_runs} of {runs} runs wrong')
    flow.deallocate()


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('captures', nargs='+', type=Path, help='a folder of solve_capture_*.npz, or the files')
    parser.add_argument('--runs', type=int, default=50, help='GPU solves per capture (default 50)')
    args = parser.parse_args()
    paths: list[Path] = []
    for item in args.captures:
        paths.extend(sorted(item.glob('solve_capture_*.npz')) if item.is_dir() else [item])
    mh.context()
    for path in paths:
        report(path, args.runs)


if __name__ == '__main__':
    main()
