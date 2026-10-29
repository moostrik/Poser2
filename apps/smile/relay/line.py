"""The waiting line of SMILE Online, and what each phone is told about the eye. Pure logic, no sockets:
the server (relay/server.py) calls it with the time and sends what it returns.

Phones join the line when they allow their camera; SMILE asks for a turn (`start_turn`) when its eye is
free, and the first in line gets it. SMILE decides every turn's start and end; here is only who waits,
who has the turn, what the eye says it is doing, and how long a phone will wait: the online turns are
timed (`durations`, their median is a typical turn, TYPICAL_TURN before any), a visitor inside takes the
same sequence, so the wait is what is left of what runs now plus a turn for everyone ahead."""

from collections import deque
from dataclasses import dataclass, field
from statistics import median

ASLEEP: str = 'asleep'                      # SMILE is not connected
LOOKING: str = 'looking for faces'          # the eye is free
INSIDE: str = 'looking at someone inside'   # a visitor in the room has the eye (SMILE's words, Online.INSIDE)
OUTSIDE: str = 'looking at someone outside' # an online turn runs
WARMING: str = 'warming up'                 # the generator is not ready (Online.WARMING)
BUSY: str = 'busy inside'                   # the operator's states that block the room (Online.BUSY)

# how far the image of a phone in line is pixelated: the block size per place, the nearest turn last
BLOCKS: tuple[int, ...] = (64, 48, 32, 16)
TYPICAL_TURN: float = 40.0                  # s an online turn takes until the relay has measured some
TURNS_TIMED: int = 10                       # the turns the typical one is the median of
LEAST_LEFT: float = 5.0                     # s: what runs now is never said to be over already


@dataclass
class Eye:
    """What SMILE last said about its eye: there, free, its status line; `laughter` the room's laughter
    switch, which the phones follow too."""
    connected: bool = False
    free: bool = False
    status: str = ASLEEP
    laughter: bool = True


@dataclass
class Line:
    waiting: list[str] = field(default_factory=list)    # phone ids, the first is next
    turn: str | None = None                             # the phone whose turn it is
    eye: Eye = field(default_factory=Eye)
    eye_since: float = 0.0                              # when the eye's status last changed
    turn_since: float = 0.0                             # when the turn started
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=TURNS_TIMED))     # the last turns, s

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
    def start_turn(self, now: float = 0.0) -> str | None:
        """SMILE asks for a turn: the first in line gets it, None when nobody waits."""
        if self.turn is not None:
            return self.turn
        if not self.waiting:
            return None
        self.turn = self.waiting.pop(0)
        self.turn_since = now
        return self.turn

    def end_turn(self, now: float = 0.0) -> str | None:
        """The turn is over (SMILE says so): returns whose it was, and remembers how long it took."""
        turn, self.turn = self.turn, None
        if turn is not None:
            self.durations.append(now - self.turn_since)
        return turn

    def typical(self) -> float:
        """How long a turn takes: the median of the last ones, TYPICAL_TURN before any."""
        return median(self.durations) if self.durations else TYPICAL_TURN

    # THE EYE
    def set_eye(self, connected: bool, free: bool = False, status: str = ASLEEP, laughter: bool = True, now: float = 0.0) -> None:
        if not connected:
            self.turn = None
            eye = Eye(False, False, ASLEEP, self.eye.laughter)
        else:
            eye = Eye(True, free, status or LOOKING, laughter)
        if (eye.connected, eye.status) != (self.eye.connected, self.eye.status):
            self.eye_since = now
        self.eye = eye

    def reason(self) -> str:
        """Why a phone waits: `outside` (an online turn), else the eye's `free`, `inside`, `warming`, `busy`."""
        if self.turn is not None:
            return 'outside'
        if self.eye.free or self.eye.status == LOOKING:
            return 'free'
        return {INSIDE: 'inside', WARMING: 'warming', BUSY: 'busy'}.get(self.eye.status, 'busy')

    def wait(self, ahead: int, now: float) -> float | None:
        """Seconds until a phone with `ahead` phones before it gets the eye: what is left of the turn or
        the visitor inside running now (never under LEAST_LEFT), plus a typical turn per phone ahead;
        None while nobody can say (the generator warming up, the room busy)."""
        reason: str = self.reason()
        typical: float = self.typical()
        if reason == 'outside':
            left: float = max(LEAST_LEFT, typical - (now - self.turn_since))
        elif reason == 'inside':
            left = max(LEAST_LEFT, typical - (now - self.eye_since))
        elif reason == 'free':
            left = LEAST_LEFT
        else:
            return None
        return left + ahead * typical

    def state(self, phone: str, now: float = 0.0) -> dict:
        """What a phone is told: `eye` is `asleep`, `you`, `waiting`, `busy` or `free`; `status` the
        line under the image; `place` and `waiting` the line; `block` the pixel size while waiting;
        `reason` why it waits (`reason()`) and `eta` the seconds until its turn (`wait()`, None unknown)."""
        waiting: int = len(self.waiting)
        laughter: bool = self.eye.laughter
        if not self.eye.connected:
            return {'type': 'state', 'eye': 'asleep', 'status': ASLEEP, 'place': self.place(phone), 'waiting': waiting, 'block': BLOCKS[0],
                    'laughter': laughter, 'reason': 'asleep', 'eta': None}
        if self.turn == phone:
            return {'type': 'state', 'eye': 'you', 'status': '', 'place': 0, 'waiting': waiting, 'block': 1, 'laughter': laughter,
                    'reason': 'you', 'eta': None}
        status: str = OUTSIDE if self.turn is not None else self.eye.status
        reason: str = self.reason()
        place: int = self.place(phone)
        if place:
            if place > 1:
                status = f'{place - 1} before you, {status}'
            return {'type': 'state', 'eye': 'waiting', 'status': status, 'place': place, 'waiting': waiting,
                    'block': BLOCKS[max(0, len(BLOCKS) - place)], 'laughter': laughter, 'reason': reason, 'eta': self.wait(place - 1, now)}
        eye: str = 'free' if self.eye.free and self.turn is None else 'busy'
        return {'type': 'state', 'eye': eye, 'status': status, 'place': 0, 'waiting': waiting, 'block': BLOCKS[0], 'laughter': laughter,
                'reason': reason, 'eta': self.wait(waiting, now)}
