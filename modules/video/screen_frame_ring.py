"""ScreenFrameRing: the screen recorder's frames in plain RAM, in blocks added one at a time."""

from threading import Lock

import numpy as np


class ScreenFrameRing:
    """Slots of one frame shape, in numpy blocks of plain (pageable) RAM: a block allocates without locking any page,
    and the memory goes back when the ring is cleared. Blocks are added and the ring cleared by one thread; a slot
    is read on any."""

    def __init__(self) -> None:
        self._lock: Lock = Lock()
        self._blocks: list[np.ndarray] = []     # each (slots, *shape) uint8
        self._shape: tuple[int, ...] = (0,)

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
        return int(np.prod(self._shape))

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(block.nbytes for block in self._blocks)

    def clear(self, shape: tuple[int, ...] | None = None) -> None:
        """Let every block go; with `shape`, the frames to come have that shape."""
        with self._lock:
            self._blocks = []
            if shape is not None:
                self._shape = shape

    def add_block(self, slots: int) -> None:
        """Grow the ring by `slots` frames; raises MemoryError when the system has no room."""
        block = np.empty((slots, *self._shape), np.uint8)
        with self._lock:
            self._blocks.append(block)

    def frame(self, slot: int) -> np.ndarray:
        """Slot `slot`, counted over the blocks in the order they were added."""
        with self._lock:
            for block in self._blocks:
                if slot < block.shape[0]:
                    return block[slot]
                slot -= block.shape[0]
        raise IndexError('screen frame slot outside the ring')
