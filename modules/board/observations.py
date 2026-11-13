from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.tracker import Tracklet


class HasObservations(Protocol):
    """Every live per-camera observation from the panoramic tracker, one per camera that sees a player: the unfused
    view, beside ``HasTracklets``' fused one; read by the calibration display."""
    def get_observations(self) -> list[Tracklet]: ...
    def set_observations(self, observations: list[Tracklet]) -> None: ...


class ObservationStoreMixin:
    """Thread-safe storage for the tracker's per-camera observations."""

    def __init__(self) -> None:
        self._observation_lock = Lock()
        self._observations: list[Tracklet] = []

    def get_observations(self) -> list[Tracklet]:
        with self._observation_lock:
            return list(self._observations)

    def set_observations(self, observations: list[Tracklet]) -> None:
        with self._observation_lock:
            self._observations = list(observations)
