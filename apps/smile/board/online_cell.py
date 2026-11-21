from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..player import OnlineCellMode


@dataclass(frozen=True)
class OnlineCell:
    """SMILE Online's cell of the walls: the wall, its cell, what it shows (`QR`, the card with the code to `page`, or
    `SCREEN`, the main screen as the phone sees it) and the phone page."""
    wall: int
    cell: int
    mode: OnlineCellMode
    page: str


class HasOnlineCell(Protocol):
    """The online cell; None while it plays a clip like any cell."""
    def get_online_cell(self) -> OnlineCell | None: ...
    def set_online_cell(self, cell: OnlineCell | None) -> None: ...


class OnlineCellStoreMixin:
    """Thread-safe storage of the online cell."""

    def __init__(self) -> None:
        self._online_cell_lock = Lock()
        self._online_cell: OnlineCell | None = None

    def get_online_cell(self) -> OnlineCell | None:
        with self._online_cell_lock:
            return self._online_cell

    def set_online_cell(self, cell: OnlineCell | None) -> None:
        with self._online_cell_lock:
            self._online_cell = cell
