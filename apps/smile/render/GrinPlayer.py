"""Buffer between the generator and the render: the grin frames, played on the render's clock.

The generator adds the frames of a grin (a clip) as ComfyUI makes them; the render asks once per
render frame which frame is due, so every frame shows equally long whatever the generator's timing.

A clip shows its first frame at once and holds it: the sequence is still DIFFUSION then, so the
render blends it over the portrait as one more generation step. The clip plays when the hold is over
(settings.grin_delay, counted from the portrait) and, at the speed the frames come in, the last one will
be there before its turn. A grin can't end before ComfyUI has made its last frame, so that wait
costs no time: it only moves the pause to the front.
"""

import time
from collections import deque
from threading import Lock

import numpy as np

from apps.smile.render.settings import SequencerSettings

MIN_FRAMES: int = 8     # frames needed to measure how fast they come in
RATE_FRAMES: int = 30   # that speed is measured over the newest frames
MARGIN: float = 0.2     # spare time between the arrival of the last frame and its turn (s)


class GrinPlayer():
    def __init__(self, settings: SequencerSettings) -> None:
        self.settings: SequencerSettings = settings
        self._mutex: Lock = Lock()
        self._frames: deque[np.ndarray] = deque()
        self._open: bool = False            # a clip is being made or played
        self._ended: bool = False           # all its frames are in
        self._hold: bool = False
        self._total: int = 0
        self._added: int = 0
        self._shown: int = 0
        self._origin: float | None = None   # the hold and the waits count from here: the portrait, else the first frame
        self._first: float = 0.0            # arrival of the first frame
        self._last: float = 0.0             # arrival of the newest frame
        self._arrivals: deque[float] = deque(maxlen=RATE_FRAMES)
        self._measurable: float = 0.0       # arrival of the frame that made the speed measurable
        self._next: float | None = None     # turn of the next frame; None until the clip plays
        self.plays: int = 0                 # clips that started playing

    # GENERATOR THREAD
    def begin(self, total: int, hold: bool) -> None:
        """A new clip of *total* frames; with *hold* it waits at least settings.grin_delay on its first frame."""
        with self._mutex:
            self._reset()
            self._open = True
            self._total = max(total, 1)
            self._hold = hold

    def hold_from_now(self) -> None:
        """The portrait is shown: the hold counts from here instead of from the first frame."""
        with self._mutex:
            if self._open and self._added == 0:
                self._origin = time.time()

    def add(self, frame: np.ndarray) -> None:
        now: float = time.time()
        with self._mutex:
            if not self._open or self._ended:
                return      # no clip: the warm-up, or a frame that arrives after a reset
            if self._added == 0:
                self._first = now
                if self._origin is None:
                    self._origin = now
            self._added += 1
            self._last = now
            self._arrivals.append(now)
            if self._added == min(MIN_FRAMES, self._total):
                self._measurable = now
            self._frames.append(frame)

    def end(self) -> None:
        """All frames of the clip are in; it stays busy until the last one has been shown."""
        with self._mutex:
            if not self._open or self._ended:
                return
            self._ended = True
            if self._added == 0 or self._origin is None:
                self._open = False
                return
            if self._added < MIN_FRAMES:
                return      # a still: nothing to measure
            made_fps: float = (self._added - 1) / (self._last - self._first) if self._last > self._first else 0.0
            # the earliest start at which the last frame arrived before its turn
            needed: float = max(self._last + MARGIN - self._play_time(self._added), self._measurable)
            buffer_wait: float = max(needed - self._origin, 0.0)
        self.settings.grin_made_fps = round(made_fps, 1)
        self.settings.grin_buffer_wait = round(buffer_wait, 2)

    def clear(self) -> None:
        """Drop the clip (reset, failure)."""
        with self._mutex:
            self._reset()

    @property
    def busy(self) -> bool:
        """A clip has frames to come or to show."""
        with self._mutex:
            return self._open

    # RENDER THREAD
    def update(self, now: float) -> np.ndarray | None:
        """The frame that became due, or None while the current one stays. Call once per render frame."""
        start_wait: float | None = None
        with self._mutex:
            if not self._open or self._added == 0 or self._origin is None:
                return None
            frame: np.ndarray | None = None
            interval: float = 1.0 / self._fps()
            if self._shown == 0:
                frame = self._frames.popleft()
                self._shown = 1
            if self._next is None:
                if not self._ready(now):
                    return frame
                # the held first frame has had its time: the second is due at the start
                self._next = now if frame is None else now + interval
                self.plays += 1
                if self._total >= MIN_FRAMES:
                    start_wait = now - self._origin

            while self._frames and now >= self._next:
                # several due at once (a slow render frame): the newest, the clock stays steady
                frame = self._frames.popleft()
                self._shown += 1
                self._next += interval
            if not self._frames:
                if self._ended:
                    self._open = False
                elif now > self._next:
                    self._next = now    # ran dry: go on with the next frame when it arrives, without catching up in a burst
        if start_wait is not None:
            self.settings.grin_start_wait = round(start_wait, 2)
        return frame

    @property
    def playing(self) -> bool:
        """The clip has started playing (and may have finished)."""
        with self._mutex:
            return self._next is not None

    @property
    def progress(self) -> float:
        with self._mutex:
            return min(self._shown / self._total, 1.0) if self._total else 0.0

    # INTERNAL (mutex held)
    def _reset(self) -> None:
        self._frames.clear()
        self._open = False
        self._ended = False
        self._hold = False
        self._total = 0
        self._added = 0
        self._shown = 0
        self._origin = None
        self._first = self._last = self._measurable = 0.0
        self._arrivals.clear()
        self._next = None

    def _fps(self) -> float:
        return max(self.settings.grin_fps, 1.0)

    def _rate(self) -> float:
        """Frames per second as the newest frames came in; 0 when that can't be told yet.
        Not from the first frame on: the first ones come slower, and that pause is already behind us."""
        if len(self._arrivals) < 2 or self._arrivals[-1] <= self._arrivals[0]:
            return 0.0
        return (len(self._arrivals) - 1) / (self._arrivals[-1] - self._arrivals[0])

    def _play_time(self, frames: int) -> float:
        """From the start of the playback to the turn of the last frame: the second frame is due at the start."""
        return max(frames - 2, 0) / self._fps()

    def _ready(self, now: float) -> bool:
        assert self._origin is not None
        if self._hold and now - self._origin < self.settings.grin_delay:
            return False
        if self._ended or self._added >= self._total:
            return True
        if self._added < MIN_FRAMES:
            return False
        rate: float = self._rate()
        if rate <= 0.0:
            return True     # they came in all at once
        # start when the last frame, at this speed, will be in before its turn
        last_arrival: float = self._last + (self._total - self._added) / rate
        return now + self._play_time(self._total) >= last_arrival + MARGIN
