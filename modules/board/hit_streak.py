from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HitStreak:
    """The hits' sync, published per light tick (pure data; the producer defines the semantics — for
    white_space: hit = a player was crossed by the playhead this tick; hits = how many of the most recent
    hits, within one round, struck alike poses; distance = the largest posture distance, in degrees, between
    those hits, 0 with fewer than two)."""
    hit:      bool  = False
    hits:     int   = 0
    distance: float = 0.0


class HasHitStreak(Protocol):
    """Access to the latest hit streak."""
    def get_hit_streak(self) -> HitStreak: ...
    def set_hit_streak(self, streak: HitStreak) -> None: ...


class HitStreakStoreMixin:
    """Thread-safe storage of the latest hit streak."""

    def __init__(self) -> None:
        self._hit_streak_lock = Lock()
        self._hit_streak = HitStreak()

    def get_hit_streak(self) -> HitStreak:
        with self._hit_streak_lock:
            return self._hit_streak

    def set_hit_streak(self, streak: HitStreak) -> None:
        with self._hit_streak_lock:
            self._hit_streak = streak
