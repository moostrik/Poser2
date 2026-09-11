from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.tracker import Tracklet


class HasObservations(Protocol):
    """Every live per-camera observation from the panoramic tracker — one per camera that can
    see a person, not one per person.

    Distinct from ``HasTracklets``, which carries the fused result: one primary per world, the
    show's input. This channel is the *unfused* view, and it exists because the fusion is
    exactly what hides camera error: a person on a seam has two observations with two world
    angles, and if the geometry is wrong those two disagree. Nothing in the show reads this —
    it is for the calibration display.
    """
    def get_observations(self) -> list[Tracklet]: ...
    def set_observations(self, observations: list[Tracklet]) -> None: ...


class ObservationStoreMixin:
    """Thread-safe storage for the tracker's per-camera observations."""

    def __init__(self) -> None:
        self._observation_store_lock = Lock()
        self._observations: list[Tracklet] = []

    def get_observations(self) -> list[Tracklet]:
        with self._observation_store_lock:
            return list(self._observations)

    def set_observations(self, observations: list[Tracklet]) -> None:
        with self._observation_store_lock:
            self._observations = list(observations)
