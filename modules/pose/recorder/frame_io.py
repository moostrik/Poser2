"""HDF5 serialization helpers for pose frame chunks.

File layout (``pose_NNN.h5``):

    /attrs
        recording_start  float64   absolute time.time() when recording began
        chunk_start      float64   absolute time.time() when this chunk began
        chunk_index      uint32    0-based index of this chunk

    /timestamps                   float64(n_frames,)

    /tracks/{track_id}
        /attrs
            cam_id       int32     camera source for this track
        /present                  bool(n_frames,)
        /cam_ids                  int32(n_frames,)   -1 when absent
        /features/{FeatureName}
            values                float32(n_frames, ...)   NaN when absent
            scores                float32(n_frames, ...)   0.0 when absent
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from ..frame import FrameDict
    from ..features import BaseFeature

import logging
logger = logging.getLogger(__name__)


def write_chunk(
    path: Path,
    entries: list[tuple[float, 'FrameDict']],
    recording_start: float,
    chunk_start: float,
    chunk_index: int,
    feature_types: list[type['BaseFeature']],
) -> None:
    """Serialize a list of (timestamp, FrameDict) entries to an HDF5 file.

    Args:
        path:             Destination .h5 file path.
        entries:          Ordered list of (timestamp, FrameDict) captured during the chunk.
        recording_start:  Absolute time.time() when the overall recording began.
        chunk_start:      Absolute time.time() when this chunk began.
        chunk_index:      0-based chunk index (matches the suffix in the filename).
        feature_types:    Feature classes to include; others are ignored.
    """
    if not entries:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(entries)
    timestamps = np.array([ts for ts, _ in entries], dtype=np.float64)

    # Collect all track IDs seen across all frames
    track_ids: set[int] = set()
    for _, fd in entries:
        track_ids.update(fd.keys())

    # Pre-compute NaN shapes from a dummy instance of each feature type
    feature_shapes: dict[type, tuple[tuple, tuple]] = {}
    for ft in feature_types:
        dummy = ft.create_dummy()
        feature_shapes[ft] = (dummy.values.shape, dummy.scores.shape)

    with h5py.File(path, 'w') as f:
        f.attrs['recording_start'] = recording_start
        f.attrs['chunk_start'] = chunk_start
        f.attrs['chunk_index'] = chunk_index

        f.create_dataset('timestamps', data=timestamps)

        tracks_grp = f.create_group('tracks')

        for track_id in sorted(track_ids):
            t_grp = tracks_grp.create_group(str(track_id))

            present = np.array([track_id in fd for _, fd in entries], dtype=bool)
            cam_ids = np.array(
                [fd[track_id].cam_id if track_id in fd else -1 for _, fd in entries],
                dtype=np.int32,
            )
            t_grp.create_dataset('present', data=present)
            t_grp.create_dataset('cam_ids', data=cam_ids)

            feat_grp = t_grp.create_group('features')
            for ft in feature_types:
                val_shape, score_shape = feature_shapes[ft]

                all_values = np.full((n_frames, *val_shape), np.nan, dtype=np.float32)
                all_scores = np.zeros((n_frames, *score_shape), dtype=np.float32)

                for i, (_, fd) in enumerate(entries):
                    if track_id in fd:
                        frame = fd[track_id]
                        if ft in frame:
                            feature = frame[ft]
                            all_values[i] = feature.values
                            all_scores[i] = feature.scores

                ft_grp = feat_grp.create_group(ft.__name__)
                ft_grp.create_dataset('values', data=all_values, compression='gzip', compression_opts=4)
                ft_grp.create_dataset('scores', data=all_scores, compression='gzip', compression_opts=4)


def read_chunk(path: Path) -> dict:
    """Deserialize a pose chunk HDF5 file.

    Returns a dict::

        {
            'recording_start': float,
            'chunk_start':     float,
            'chunk_index':     int,
            'timestamps':      np.ndarray,          # float64(n_frames,)
            'tracks': {
                track_id: {
                    'present':  np.ndarray,          # bool(n_frames,)
                    'cam_ids':  np.ndarray,          # int32(n_frames,)
                    'features': {
                        'BBox': {'values': np.ndarray, 'scores': np.ndarray},
                        ...
                    }
                }
            }
        }
    """
    with h5py.File(path, 'r') as f:
        attrs = f.attrs
        result: dict = {
            'recording_start': float(attrs['recording_start']),   # type: ignore[arg-type]
            'chunk_start':     float(attrs['chunk_start']),        # type: ignore[arg-type]
            'chunk_index':     int(attrs['chunk_index']),          # type: ignore[arg-type]
            'timestamps':      f['timestamps'][:],                 # type: ignore[index]
            'tracks':          {},
        }
        tracks_grp = f['tracks']                                   # type: ignore[index]
        for track_id_str, t_grp in tracks_grp.items():            # type: ignore[union-attr]
            track_id = int(track_id_str)
            track_data: dict = {
                'present':  t_grp['present'][:],
                'cam_ids':  t_grp['cam_ids'][:],
                'features': {},
            }
            for feat_name, feat_grp in t_grp['features'].items():
                track_data['features'][feat_name] = {
                    'values': feat_grp['values'][:],
                    'scores': feat_grp['scores'][:],
                }
            result['tracks'][track_id] = track_data

    return result
