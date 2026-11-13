"""Where a player moves in a recording, from its metadata alone: per second, how far the first tracklet's box center
travels (in the frame's widths and heights, summed), so a probe can be run over the stretch it needs.

    python scripts/recording_motion.py FOLDER [--app deep_flow]
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.recording import read_metadata  # noqa: E402


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--app', default='deep_flow')
    args = parser.parse_args()
    folder: Path = Path(__file__).resolve().parent.parent / 'apps' / args.app / 'data' / 'recordings' / args.folder
    frames = read_metadata(folder / '0_000.mkv').frames
    t0: float = frames[0].timestamp
    centers: list[tuple[float, float, float]] = []
    for frame in frames:
        if frame.tracklets:
            roi = frame.tracklets[0].roi
            x, y, w, h = (roi.x, roi.y, roi.width, roi.height) if hasattr(roi, 'x') else roi
            centers.append((frame.timestamp - t0, x + w / 2.0, y + h / 2.0))
    if len(centers) < 2:
        print('no player in the recording')
        return 1
    c: np.ndarray = np.array(centers)
    for second in range(int(c[-1, 0]) + 1):
        part: np.ndarray = c[(c[:, 0] >= second) & (c[:, 0] < second + 1)]
        if len(part) < 2:
            continue
        travel: float = float(np.abs(np.diff(part[:, 1:], axis=0)).sum())
        print(f'{second:3d} s  {travel:6.3f}  ' + '#' * int(travel * 40))
    return 0


if __name__ == '__main__':
    sys.exit(main())
