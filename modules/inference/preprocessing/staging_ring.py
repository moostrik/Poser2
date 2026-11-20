"""Pinned staging buffers that frames are copied from to the GPU, a ring of `STAGING_COUNT` per key; a buffer is
reused once the copy queued from it has run, its event passed."""

from collections.abc import Hashable

import torch

STAGING_COUNT: int = 3      # pinned buffers per key: copies the GPU may be behind before the caller waits


class StagingRing:
    """See the module docstring. Used from one thread."""

    def __init__(self) -> None:
        # Each key's pinned buffers, the events of the copies queued from them, the one used last
        self._rings: dict[Hashable, tuple[list[torch.Tensor], list[torch.cuda.Event | None], int]] = {}

    def buffer(self, key: Hashable, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """The key's next pinned buffer, (re)made at `shape` and `dtype`; waits for the copy last queued from it."""
        events: list[torch.cuda.Event | None]
        buffers, events, slot = self._rings.get(key, ([], [], -1))
        if not buffers or tuple(buffers[0].shape) != shape or buffers[0].dtype != dtype:
            buffers = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(STAGING_COUNT)]
            events = [None] * STAGING_COUNT
        slot = (slot + 1) % STAGING_COUNT
        event: torch.cuda.Event | None = events[slot]
        if event is not None:
            event.synchronize()
        self._rings[key] = (buffers, events, slot)
        return buffers[slot]

    def queued(self, key: Hashable, event: torch.cuda.Event) -> None:
        """The copy from the key's last buffer queued; the buffer is free again once `event` has passed."""
        _, events, slot = self._rings[key]
        events[slot] = event
