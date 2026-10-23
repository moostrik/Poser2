# Standard library imports
from collections import deque


class PlayerSlots:
    """Assigns live worlds to the player slots 0..max_players-1 everything downstream is wired for.

    A world gets a slot when one is free — the oldest waiting world first; world ids are handed out
    in order — and keeps it for life, LOST included. The slot frees when the world is removed, and
    freed slots are reused as late as possible (FIFO), so a new person never inherits a just-departed
    person's downstream state. Owned by `ObservationStore` and mutated only from its world lifecycle,
    on the tracker thread.
    """

    def __init__(self, max_players: int) -> None:
        self._free: deque[int] = deque(range(max_players))
        self._slot_for: dict[int, int] = {}   # world id -> slot
        self._waiting: list[int] = []         # slotless world ids, oldest first

    def slot_of(self, world_id: int) -> int | None:
        """The slot this world is emitted on, or None while it waits."""
        return self._slot_for.get(world_id)

    def add(self, world_id: int) -> None:
        """A world was created: give it a slot if one is free, else it waits."""
        if self._free:
            self._slot_for[world_id] = self._free.popleft()
        else:
            self._waiting.append(world_id)

    def remove(self, world_id: int) -> None:
        """A world died: its slot goes to the back of the queue, then the oldest waiter is served."""
        slot: int | None = self._slot_for.pop(world_id, None)
        if slot is None:
            self._waiting.remove(world_id)
            return
        self._free.append(slot)
        self._promote()

    def merge(self, keep_id: int, drop_id: int) -> None:
        """Two worlds turned out to be one person: `keep_id` survives, slotted if either was.

        A waiting `keep_id` inherits `drop_id`'s slot, so a person already on the channel stays on
        it through a split repair — even past an older waiter: continuity beats queue fairness.
        """
        drop_slot: int | None = self._slot_for.pop(drop_id, None)
        if drop_slot is None:
            self._waiting.remove(drop_id)
        elif keep_id in self._slot_for:
            self._free.append(drop_slot)
            self._promote()
        else:
            self._waiting.remove(keep_id)
            self._slot_for[keep_id] = drop_slot

    def _promote(self) -> None:
        while self._free and self._waiting:
            self._slot_for[self._waiting.pop(0)] = self._free.popleft()

    @property
    def in_use(self) -> int:
        return len(self._slot_for)

    @property
    def waiting(self) -> int:
        return len(self._waiting)
