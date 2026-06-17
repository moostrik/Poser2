"""Playhead hit detection — shared geometry and event type for the rotating playhead."""

from dataclasses import dataclass

import numpy as np

from modules.tracker import Tracklet
from modules.pose.frame import Frame
from modules.pose import features


@dataclass(frozen=True)
class PlayheadHit:
    """Emitted when the rotating playhead sweeps past an active player."""
    track_id: int
    position: float  # azimuth 0.0–1.0 of the person at moment of crossing


def sweep_contains(prev: float, curr: float, pos: float) -> bool:
    """Return True if *pos* falls inside (prev, curr] on the unit circle, with wrap-around."""
    if prev <= curr:
        return prev < pos <= curr
    else:
        return pos > prev or pos <= curr


def detect_hits(
    frames: list[Frame],
    tracklets: dict[int, Tracklet],
    prev: float,
    curr: float,
) -> tuple[PlayheadHit, ...]:
    """Return one PlayheadHit for each active player whose azimuth falls in (prev, curr]."""
    hits: list[PlayheadHit] = []
    for frame in frames:
        tracklet = tracklets.get(frame.track_id)
        if tracklet is None or not tracklet.is_active:
            continue
        pos: float = frame[features.Azimuth].value
        if not np.isnan(pos) and sweep_contains(prev, curr, pos):
            hits.append(PlayheadHit(track_id=frame.track_id, position=pos))
    return tuple(hits)
