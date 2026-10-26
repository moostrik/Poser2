"""The waiting line of SMILE Online, and what each phone is told about the eye. Pure logic, no sockets:
the server (relay/server.py) calls it and sends what it returns.

Phones join the line when they allow their camera; SMILE asks for a turn (`start_turn`) when its eye is
free, and the first in line gets it. SMILE decides every turn's start and end; here is only who waits,
who has the turn, and what the eye says it is doing."""

from dataclasses import dataclass, field

ASLEEP: str = 'asleep'                      # SMILE is not connected
LOOKING: str = 'looking for faces'          # the eye is free
OUTSIDE: str = 'looking at someone outside' # an online turn runs

# how far the image of a phone in line is pixelated: the block size per place, the nearest turn last
BLOCKS: tuple[int, ...] = (64, 48, 32, 16)


@dataclass
class Eye:
    """What SMILE last said about its eye."""
    connected: bool = False
    free: bool = False
    status: str = ASLEEP


@dataclass
class Line:
    waiting: list[str] = field(default_factory=list)    # phone ids, the first is next
    turn: str | None = None                             # the phone whose turn it is
    eye: Eye = field(default_factory=Eye)

    # THE LINE
    def join(self, phone: str) -> int:
        """Join at the back (or stay where it is); returns the place, 1 = next."""
        if phone != self.turn and phone not in self.waiting:
            self.waiting.append(phone)
        return self.place(phone)

    def leave(self, phone: str) -> bool:
        """Leave the line, or the turn. Returns whether it was the turn that ended."""
        if phone in self.waiting:
            self.waiting.remove(phone)
        if self.turn == phone:
            self.turn = None
            return True
        return False

    def place(self, phone: str) -> int:
        """1 for the next in line, 0 for the turn or a phone not in line."""
        return self.waiting.index(phone) + 1 if phone in self.waiting else 0

    # THE TURN
    def start_turn(self) -> str | None:
        """SMILE asks for a turn: the first in line gets it, None when nobody waits."""
        if self.turn is not None:
            return self.turn
        if not self.waiting:
            return None
        self.turn = self.waiting.pop(0)
        return self.turn

    def end_turn(self) -> str | None:
        """The turn is over (SMILE says so): returns whose it was."""
        turn, self.turn = self.turn, None
        return turn

    # THE EYE
    def set_eye(self, connected: bool, free: bool = False, status: str = ASLEEP) -> None:
        if not connected:
            self.turn = None
            self.eye = Eye(False, False, ASLEEP)
        else:
            self.eye = Eye(True, free, status or LOOKING)

    def state(self, phone: str) -> dict:
        """What a phone is told: `eye` is `asleep`, `you`, `waiting`, `busy` or `free`; `status` the
        line under the image; `place` and `waiting` the line; `block` the pixel size while waiting."""
        waiting: int = len(self.waiting)
        if not self.eye.connected:
            return {'type': 'state', 'eye': 'asleep', 'status': ASLEEP, 'place': self.place(phone), 'waiting': waiting, 'block': BLOCKS[0]}
        if self.turn == phone:
            return {'type': 'state', 'eye': 'you', 'status': '', 'place': 0, 'waiting': waiting, 'block': 1}
        status: str = OUTSIDE if self.turn is not None else self.eye.status
        place: int = self.place(phone)
        if place:
            if place > 1:
                status = f'{place - 1} before you, {status}'
            return {'type': 'state', 'eye': 'waiting', 'status': status, 'place': place, 'waiting': waiting,
                    'block': BLOCKS[max(0, len(BLOCKS) - place)]}
        eye: str = 'free' if self.eye.free and self.turn is None else 'busy'
        return {'type': 'state', 'eye': eye, 'status': status, 'place': 0, 'waiting': waiting, 'block': BLOCKS[0]}
