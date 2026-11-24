"""ScreenFrameStore: the screen recorder's frames in plain RAM, in blocks added one at a time."""

from threading import Lock

import numpy as np


class ScreenFrameStore:
    """Slots of one frame shape, in numpy blocks of plain RAM added one at a time; a slot is read on any thread."""

    def __init__(self) -> None:
        self._lock: Lock = Lock()
        self._blocks: list[np.ndarray] = []     # each (slots, *shape) of dtype
        self._shape: tuple[int, ...] = (0,)
        self._dtype: np.dtype = np.dtype(np.uint8)

    @property
    def num_slots(self) -> int:
        with self._lock:
            return sum(block.shape[0] for block in self._blocks)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one frame."""
        return self._shape

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self._shape)) * self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(block.nbytes for block in self._blocks)

    def clear(self, shape: tuple[int, ...] | None = None, dtype: type[np.unsignedinteger] = np.uint8) -> None:
        """Let every block go; with `shape`, the frames to come have that shape and `dtype`."""
        with self._lock:
            self._blocks = []
            if shape is not None:
                self._shape = shape
                self._dtype = np.dtype(dtype)

    def add_block(self, slots: int) -> None:
        """Grow the store by `slots` frames; raises MemoryError when the system has no room."""
        block = np.empty((slots, *self._shape), self._dtype)
        with self._lock:
            self._blocks.append(block)

    def frame(self, slot: int) -> np.ndarray:
        """Slot `slot`, counted over the blocks in the order they were added."""
        with self._lock:
            for block in self._blocks:
                if slot < block.shape[0]:
                    return block[slot]
                slot -= block.shape[0]
        raise IndexError('screen frame slot outside the store')
