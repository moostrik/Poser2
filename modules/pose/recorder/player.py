"""Player — loads and queries recorded pose HDF5 files.

Typical usage::

    player = Player()
    player.load(Path("recordings/20260409-001336_3_square_color_no_id"))

    # Nearest frame to a given timestamp
    frame_dict = player.get_frame_dict(some_timestamp)

    # Stream all chunks frame-by-frame
    for chunk_entries in player.iter_chunks():
        for ts, frame_dict in chunk_entries:
            ...
"""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np

from ..frame import Frame, FrameDict
from ..features import FEATURES
from .frame_io import read_chunk

import logging
logger = logging.getLogger(__name__)


class Player:
    """Loads recorded ``pose_NNN.h5`` files and reconstructs FrameDict objects."""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._all_timestamps: np.ndarray = np.array([], dtype=np.float64)
        self._chunk_offsets: list[int] = []
        self._feature_by_name: dict[str, type] = {ft.__name__: ft for ft in FEATURES}

    def load(self, folder: Path) -> None:
        """Discover and load all ``pose_*.h5`` files in *folder*."""
        chunk_files = sorted(folder.glob('pose_*.h5'))
        if not chunk_files:
            logger.warning("Player: no pose_*.h5 files in %s", folder)
        self._chunks = [read_chunk(f) for f in chunk_files]

        timestamps_list: list[np.ndarray] = []
        self._chunk_offsets = []
        offset = 0
        for chunk in self._chunks:
            self._chunk_offsets.append(offset)
            timestamps_list.append(chunk['timestamps'])
            offset += len(chunk['timestamps'])

        self._all_timestamps = (
            np.concatenate(timestamps_list) if timestamps_list else np.array([], dtype=np.float64)
        )
        logger.info("Player loaded %d chunks, %d frames from %s",
                    len(self._chunks), len(self._all_timestamps), folder)

    def get_frame_dict(self, timestamp: float) -> FrameDict:
        """Return the FrameDict at the stored timestamp nearest to *timestamp*."""
        if len(self._all_timestamps) == 0:
            return {}

        idx = bisect_left(self._all_timestamps, timestamp)
        if idx >= len(self._all_timestamps):
            idx = len(self._all_timestamps) - 1
        elif idx > 0:
            # Pick whichever neighbour is closer
            if abs(self._all_timestamps[idx - 1] - timestamp) < abs(self._all_timestamps[idx] - timestamp):
                idx -= 1

        chunk_idx = self._chunk_index_for(idx)
        local_idx = idx - self._chunk_offsets[chunk_idx]
        return self._reconstruct(self._chunks[chunk_idx], local_idx)

    def iter_chunks(self):
        """Yield ``list[tuple[float, FrameDict]]`` for each recorded chunk."""
        for chunk in self._chunks:
            entries: list[tuple[float, FrameDict]] = []
            for frame_idx, ts in enumerate(chunk['timestamps']):
                entries.append((float(ts), self._reconstruct(chunk, frame_idx)))
            yield entries

    # ── Internal helpers ─────────────────────────────────────────────────

    def _chunk_index_for(self, global_idx: int) -> int:
        """Return the chunk index that contains *global_idx*."""
        for i in range(len(self._chunk_offsets) - 1):
            if self._chunk_offsets[i + 1] > global_idx:
                return i
        return len(self._chunk_offsets) - 1

    def _reconstruct(self, chunk: dict, frame_idx: int) -> FrameDict:
        result: FrameDict = {}
        ts = float(chunk['timestamps'][frame_idx])

        for track_id, track_data in chunk['tracks'].items():
            if not track_data['present'][frame_idx]:
                continue
            cam_id = int(track_data['cam_ids'][frame_idx])
            features: dict = {}
            for feat_name, arrays in track_data['features'].items():
                ft = self._feature_by_name.get(feat_name)
                if ft is None:
                    continue
                values = arrays['values'][frame_idx].astype(np.float32)
                scores = arrays['scores'][frame_idx].astype(np.float32)
                features[ft] = ft(values, scores)
            result[track_id] = Frame(track_id, cam_id, ts, features)

        return result
