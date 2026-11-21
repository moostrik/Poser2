from typing import Generic, TypeVar

T = TypeVar('T')


class DataCache(Generic[T]):
    """Caches data and tracks changes."""

    def __init__(self):
        self._cached: T | None = None
        self._changed: bool = False
        self._lost: bool = False

    def set(self, new_data: T | None) -> None:
        if new_data is self._cached:
            self._changed = False
            self._lost = False
            return

        self._lost = new_data is None and self._cached is not None
        self._changed = True
        self._cached = new_data

    @property
    def data(self) -> T | None:
        return self._cached

    @property
    def changed(self) -> bool:
        return self._changed

    @property
    def idle(self) -> bool:
        return not self._changed

    @property
    def lost(self) -> bool:
        return self._lost

    @property
    def has_data(self) -> bool:
        return self._cached is not None

    @property
    def empty(self) -> bool:
        return self._cached is None
