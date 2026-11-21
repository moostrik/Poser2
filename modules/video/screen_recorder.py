"""ScreenRecorder: records the screen frames the screen capture hands over into RAM, holds a take, and
encodes the parts asked for to HEVC mp4s.

The capture (main thread) calls `submit_screen_frame` with each finished frame; the recorder's own thread copies
it into its store (`ScreenFrameStore`, plain RAM, grown by one-second blocks) and gives the capture's buffer back. The
render tick (`update`) runs the states: it says whether to capture this tick (`capturing`) and at which size
(`capture_format`), and acts on the buttons, which only set Events. Design and reasons: modules/video/VIDEO.md.
"""

import logging
import time
from bisect import bisect_left
from collections import deque
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Callable, Protocol

import numpy as np
import psutil

from modules.settings import BaseSettings, Field, Group, Widget

from .clip_writer import ClipInput, ClipWriter, master_encode
from .screen_format import BitDepth, Chroma, ScreenFormat
from .screen_frame import ScreenFrame
from .screen_frame_store import ScreenFrameStore

logger = logging.getLogger(__name__)

GB: int = 1 << 30
DEFAULT_FPS: float = 60.0       # ticks/s assumed until the ticks are measured
FPS_SMOOTH: float = 0.05        # share of a new tick interval in the measured rate
BLOCK_AHEAD: float = 0.5        # share of a block left when the next one is added
LOOK_BACK_MARGIN: float = 1.2   # the look-back ring's frames over `duration` at the rate measured, for faster ticks
QUEUE_LIMIT: int = 3            # frames waiting for the recorder's thread; one more is dropped, the oldest
POLL: float = 0.05              # s the thread waits for work before it looks again
ESTIMATE_INTERVAL: float = 1.0  # s between estimates of `max_duration`
SHOW_INTERVAL: float = 0.25     # s between updates of the readouts


class ScreenSize(Enum):
    """A recording size preset, landscape (width, height); portrait swaps them."""
    HD =     (1920, 1080)
    QHD =    (2560, 1440)
    UHD =    (3840, 2160)
    CUSTOM = (0, 0)


class ScreenRecorderState(Enum):
    IDLE =      auto()
    RECORDING = auto()
    HELD =      auto()
    ENCODING =  auto()
    ERROR =     auto()


class ScreenRecorderFrameSettings(BaseSettings):
    main_window:  Field[bool]                = Field(False, widget=Widget.switch, description="Record the main window instead of the sources")
    size:         Field[ScreenSize]          = Field(ScreenSize.UHD, description="Recording size preset", access=Field.READWRITE)
    portrait:     Field[bool]                = Field(True, widget=Widget.switch, description="Upright: width and height swapped", access=Field.READWRITE)
    width:        Field[int]                 = Field(2160, min=2, max=8192, step=2, description="Recording width (px); a change sets the preset to CUSTOM", access=Field.READWRITE)
    height:       Field[int]                 = Field(3840, min=2, max=8192, step=2, description="Recording height (px); a change sets the preset to CUSTOM", access=Field.READWRITE)
    snap:         Field[bool]                = Field(False, widget=Widget.button, description="Set the size to the first source's")
    fill:         Field[bool]                = Field(False, widget=Widget.switch, description="Fill the frame and cut the overhang, instead of black bars")


class ScreenRecorderQualitySettings(BaseSettings):
    chroma:       Field[Chroma]              = Field(Chroma.YUV444, description="Color resolution: 4:4:4 full, 4:2:0 half (twice the seconds)")
    bit_depth:    Field[BitDepth]            = Field(BitDepth.BIT_8, description="Bits per sample: 10 keeps smooth gradients (fewer seconds)")
    frame_step:   Field[int]                 = Field(1, min=1, max=2, step=1, description="Record every tick (1), or every other (2)")
    max_duration: Field[float]               = Field(0.0, access=Field.READ, description="Longest recording the free RAM holds (s)")


class ScreenRecorderRecordSettings(BaseSettings):
    start:        Field[bool]                = Field(False, widget=Widget.button, description="Start screen recording")
    stop:         Field[bool]                = Field(False, widget=Widget.button, description="Stop screen recording and hold the take")
    look_back:    Field[bool]                = Field(False, widget=Widget.switch, description="Keep the last seconds all the time; Stop holds them")
    duration:     Field[float]               = Field(0.0, min=0.0, max=120.0, description="Recording length (s); 0 or above the maximum uses the maximum")
    ram_reserve:  Field[float]               = Field(8.0, min=1.0, max=64.0, description="RAM left free for Windows and the app (GB)")
    state:        Field[ScreenRecorderState] = Field(ScreenRecorderState.IDLE, access=Field.READ, description="Idle, recording, held, encoding")
    recorded:     Field[float]               = Field(0.0, access=Field.READ, description="Seconds in the store (s)")
    dropped:      Field[int]                 = Field(0, access=Field.READ, description="Frames of this take left out: no buffer free, or the queue full")


class ScreenRecorderTakeSettings(BaseSettings):
    playhead:     Field[list[float]]         = Field([0.0, 0.0], access=Field.READ, visible=False, description="Where the held take is looked at, and its length (s)")
    scrub:        Field[float]               = Field(-1.0, visible=False, saved=False, description="Where the dragged marker is, as a fraction of the length; -1 when none is")
    seek:         Field[float]               = Field(-1.0, visible=False, saved=False, description="Where the bar was pressed, as a fraction of the length; -1 otherwise")
    in_out:       Field[list[float]]         = Field([0.0, 1.0], widget=Widget.timeline, options=playhead, drag=scrub, seek=seek, saved=False,
                                                     label="In / Out", description="The part of the held take that save writes", access=Field.READWRITE)
    play:         Field[bool]                = Field(False, widget=Widget.switch, saved=False, description="Play the held take between in and out, looped, over the main window; the show waits", access=Field.READWRITE)
    save:         Field[bool]                = Field(False, widget=Widget.button, description="Write the held take between in and out")
    discard:      Field[bool]                = Field(False, widget=Widget.button, description="Let the held take go")
    name:         Field[str]                 = Field("", widget=Widget.input, description="Added to the file name")
    quality:      Field[int]                 = Field(18, min=0, max=51, description="Constant quality, lower is better")


class ScreenRecorderSettings(BaseSettings):
    frame:   Group[ScreenRecorderFrameSettings]   = Group(ScreenRecorderFrameSettings, description="What is recorded, at what size")
    quality: Group[ScreenRecorderQualitySettings] = Group(ScreenRecorderQualitySettings, description="The frames' format, which sets the length")
    record:  Group[ScreenRecorderRecordSettings]  = Group(ScreenRecorderRecordSettings, description="Recording into RAM")
    take:    Group[ScreenRecorderTakeSettings]    = Group(ScreenRecorderTakeSettings, description="The held take: look, play, save")


# SIZE RULES: width and height are the truth; the preset and portrait follow them

def even(value: int) -> int:
    """`value` rounded down to an even number of at least 2, as NV12 needs."""
    return max(2, value - value % 2)


def preset_size(size: ScreenSize, portrait: bool) -> tuple[int, int]:
    """(width, height) of a preset, upright when `portrait`."""
    width, height = size.value
    return (height, width) if portrait else (width, height)


def matching_preset(width: int, height: int) -> ScreenSize:
    """The preset of this size in either orientation, else CUSTOM."""
    landscape = (max(width, height), min(width, height))
    for size in ScreenSize:
        if size is not ScreenSize.CUSTOM and size.value == landscape:
            return size
    return ScreenSize.CUSTOM


def frames_for(seconds: float, fps: float, frame_step: int) -> int:
    """Frames that `seconds` hold at `fps` ticks/s, one per `frame_step` ticks."""
    return max(1, int(seconds * fps / frame_step))


def available_ram() -> int:
    """The RAM available to processes (bytes)."""
    return int(psutil.virtual_memory().available)


class ClipSink(Protocol):
    """What the encode writes the frames' YUV planes to."""
    comment: str
    def add(self, image: np.ndarray) -> None: ...
    def close(self) -> bool: ...


ClipSinkFactory = Callable[[str, float, int, ScreenFormat], ClipSink]     # (path, fps, quality, the frames' format)

_CLIP_INPUTS: dict[tuple[Chroma, BitDepth], ClipInput] = {
    (Chroma.YUV420, BitDepth.BIT_8):  ClipInput.NV12,
    (Chroma.YUV444, BitDepth.BIT_8):  ClipInput.YUV444P,
    (Chroma.YUV420, BitDepth.BIT_10): ClipInput.P010,
    (Chroma.YUV444, BitDepth.BIT_10): ClipInput.XV30,
}


def hevc_clip(path: str, fps: float, quality: int, screen_format: ScreenFormat) -> ClipSink:
    """An HEVC mp4 on the GPU from the frames' YUV planes in their chroma and bit depth."""
    clip_input = _CLIP_INPUTS[(screen_format.chroma, screen_format.bit_depth)]
    return ClipWriter(path, fps, master_encode(clip_input, quality), clip_input)


class _Take:
    """One encode: the store's slots in recorded order, their timestamps, the file, the frames' format, the frames
    left out, and whether the store is freed after it."""

    def __init__(self, slots: list[int], stamps: list[float], path: str, screen_format: ScreenFormat, dropped: int,
                 free_after: bool) -> None:
        self.slots: list[int] = slots
        self.stamps: list[float] = stamps
        self.path: str = path
        self.format: ScreenFormat = screen_format
        self.dropped: int = dropped
        self.free_after: bool = free_after


class ScreenRecorder:
    """Screen recording of the frames the screen capture hands over: Start / Stop, or look back and Stop; a take is
    held for in and out, and each save writes a part."""

    def __init__(self, settings: ScreenRecorderSettings, folder: str, clip_sink: ClipSinkFactory = hevc_clip,
                 free_memory: Callable[[], int] = available_ram) -> None:
        """`folder`: where the files go; `free_memory`: the free RAM in bytes."""
        self.settings: ScreenRecorderSettings = settings
        self._folder: Path = Path(folder)
        self._clip_sink: ClipSinkFactory = clip_sink
        self._free_memory: Callable[[], int] = free_memory
        self._store: ScreenFrameStore = ScreenFrameStore()

        self._start_requested: Event = Event()
        self._stop_requested: Event = Event()
        self._save_requested: Event = Event()
        self._discard_requested: Event = Event()
        self._exit: Event = Event()

        # shared with the thread, under the lock
        self._lock: Lock = Lock()
        self._wake: Condition = Condition(self._lock)
        self._queue: deque[ScreenFrame] = deque()
        self._accepting: bool = False       # frames go into the store
        self._take_id: int = 0              # a new one at each begin and discard: frames of an old one are let go
        self._take_start: float = 0.0       # frames from before it are let go
        self._look_back: bool = False
        self._limit: float = 0.0            # s a take spans, by its frames' times; 0: until the memory stops it
        self._target: int = 0               # slots the store grows to; 0: as many as the memory holds
        self._written: int = 0              # frames stored since the store was emptied
        self._stamps: list[float] = []      # each slot's frame's timestamp
        self._first: float | None = None    # the take's first timestamp
        self._writing: int | None = None    # the slot a frame is being copied into
        self._complete: bool = False        # the store grows no more
        self._reached: bool = False         # the duration or the memory ends the take
        self._dropped: int = 0              # frames of this take left out
        self._take: _Take | None = None     # handed to the thread to encode
        self._free: bool = False            # the thread is to free the store without an encode
        self._failed: bool = False          # the last take was not written
        self._done: Event = Event()         # the thread finished a take or a free

        # main thread only
        self._state: ScreenRecorderState = ScreenRecorderState.IDLE
        self._format: ScreenFormat = ScreenFormat(2, 2)     # the take's, set at its begin
        self._duration: float = 0.0         # the duration the look-back ring was made for
        self._frame_step: int = 1
        self._ticks: int = 0
        self._capture_now: bool = False
        self._fps: float = DEFAULT_FPS
        self._last_timestamp: float | None = None
        self._shown_at: float = 0.0
        self._error_logged: bool = False
        self._held_slots: list[int] = []    # the held take's slots, oldest first
        self._held_stamps: list[float] = [] # and their timestamps
        self._held_saved: bool = False      # a part of the held take is written
        self._saving_held: bool = False     # the thread writes a part of the held take
        self._paths: set[str] = set()       # the files handed to the thread, so no name is given twice

        # where the held take is looked at, as a fraction of its length: the dragged marker, the bar pressed or play
        self._position: float = 0.0
        self._dragging: bool = False
        self._playing: bool = False
        self._play_from: tuple[float, float] | None = None     # (tick timestamp, take moment) play counts from

        self._thread: Thread = Thread(target=self._run, name='screen recorder', daemon=True)

        self.settings.record.bind(ScreenRecorderRecordSettings.start, self._on_start)
        self.settings.record.bind(ScreenRecorderRecordSettings.stop, self._on_stop)
        self.settings.take.bind(ScreenRecorderTakeSettings.save, self._on_save)
        self.settings.take.bind(ScreenRecorderTakeSettings.discard, self._on_discard)
        self.settings.take.bind(ScreenRecorderTakeSettings.scrub, self._on_scrub)
        self.settings.take.bind(ScreenRecorderTakeSettings.seek, self._on_seek)
        self.settings.frame.bind(ScreenRecorderFrameSettings.size, self._on_size)
        self.settings.frame.bind(ScreenRecorderFrameSettings.portrait, self._on_portrait)
        self.settings.frame.bind(ScreenRecorderFrameSettings.width, self._on_width_height)
        self.settings.frame.bind(ScreenRecorderFrameSettings.height, self._on_width_height)

    @property
    def capturing(self) -> bool:
        """Whether the screen frame is to be drawn and captured this tick; main thread, after `update`."""
        return self._capture_now

    @property
    def capture_format(self) -> ScreenFormat | None:
        """The format the capture is made at while the recorder has a take, else None; main thread."""
        if self._state in (ScreenRecorderState.RECORDING, ScreenRecorderState.HELD, ScreenRecorderState.ENCODING):
            return self._format
        return None

    @property
    def fps(self) -> float:
        """The render ticks per second, measured."""
        return self._fps

    @property
    def playing(self) -> bool:
        """Whether the held take plays this tick; main thread, after `update`."""
        return self._playing

    @property
    def preview_frame(self) -> tuple[int, np.ndarray] | None:
        """The held frame under the dragged marker, or at the play position, and its slot; None when neither; render
        thread."""
        if self._state is not ScreenRecorderState.HELD or not (self._dragging or self._playing) or not self._held_stamps:
            return None
        moment = self._held_stamps[0] + self._position * self._held_length()
        index = min(bisect_left(self._held_stamps, moment), len(self._held_stamps) - 1)
        if index > 0 and moment - self._held_stamps[index - 1] < self._held_stamps[index] - moment:
            index -= 1      # the nearer of the two frames around the moment
        slot = self._held_slots[index]
        return slot, self._store.frame(slot)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the recording: a take in progress is written up to its last stored frame, and a held take none of
        whose parts is written yet is written whole; then the frames waiting are let go and the store freed."""
        self.settings.record.unbind(ScreenRecorderRecordSettings.start, self._on_start)
        self.settings.record.unbind(ScreenRecorderRecordSettings.stop, self._on_stop)
        self.settings.take.unbind(ScreenRecorderTakeSettings.save, self._on_save)
        self.settings.take.unbind(ScreenRecorderTakeSettings.discard, self._on_discard)
        self.settings.take.unbind(ScreenRecorderTakeSettings.scrub, self._on_scrub)
        self.settings.take.unbind(ScreenRecorderTakeSettings.seek, self._on_seek)
        self.settings.frame.unbind(ScreenRecorderFrameSettings.size, self._on_size)
        self.settings.frame.unbind(ScreenRecorderFrameSettings.portrait, self._on_portrait)
        self.settings.frame.unbind(ScreenRecorderFrameSettings.width, self._on_width_height)
        self.settings.frame.unbind(ScreenRecorderFrameSettings.height, self._on_width_height)
        if self._state == ScreenRecorderState.RECORDING and not self._look_back:
            self._hold()
        if self._state == ScreenRecorderState.HELD and not self._held_saved:
            self._hand_over(self._held_slots, self._held_stamps, free_after=True)
        with self._wake:
            self._exit.set()
            self._wake.notify()
        if self._thread.is_alive():
            self._thread.join()
        self._release_queue()
        self._store.clear()

    def fail(self, message: str) -> None:
        """A failure outside the recorder, as the capture's allocation: logged once, ERROR; main thread."""
        self._fail(message)

    # THE HAND-OVER (the capture's thread)

    def submit_screen_frame(self, frame: ScreenFrame) -> None:
        """Enqueue a finished frame for the recorder's thread; a frame no take wants is given back at once."""
        released: list[ScreenFrame] = []
        with self._wake:
            if not self._accepting:
                released.append(frame)
            else:
                self._queue.append(frame)
                while len(self._queue) > QUEUE_LIMIT:
                    oldest = self._queue.popleft()
                    self._dropped += 1 + oldest.skipped
                    released.append(oldest)
                self._wake.notify()
        for item in released:
            item.release()

    # BUTTONS AND SIZE RULES (any thread)

    def _on_start(self, _: bool) -> None:
        self._start_requested.set()

    def _on_stop(self, _: bool) -> None:
        self._stop_requested.set()

    def _on_save(self, _: bool) -> None:
        self._save_requested.set()

    def _on_discard(self, _: bool) -> None:
        self._discard_requested.set()

    def _on_scrub(self, value: float) -> None:
        self._dragging = value >= 0.0
        if value >= 0.0:
            self._position = value

    def _on_seek(self, value: float) -> None:
        if value >= 0.0:
            self._position = value

    def _on_size(self, size: ScreenSize) -> None:
        if size is ScreenSize.CUSTOM:
            return
        width, height = preset_size(size, self.settings.frame.portrait)
        self.settings.frame.width = width
        self.settings.frame.height = height

    def _on_portrait(self, portrait: bool) -> None:
        width, height = self.settings.frame.width, self.settings.frame.height
        if portrait != (height > width) and width != height:
            self.settings.frame.width = height
            self.settings.frame.height = width

    def _on_width_height(self, _: int) -> None:
        width, height = self.settings.frame.width, self.settings.frame.height
        self.settings.frame.size = matching_preset(width, height)
        if width != height:
            self.settings.frame.portrait = height > width

    # RENDER TICK

    def update(self, timestamp: float) -> None:
        """Act on the buttons and the thread's news, then say whether this tick is captured (`capturing`)."""
        self._measure(timestamp)
        if self._done.is_set():
            self._finish()

        if self._state == ScreenRecorderState.RECORDING:
            with self._lock:
                reached = self._reached
            if reached:
                self._hold()        # the duration or the memory ends the take
            elif self._look_back and (not self.settings.record.look_back or self._format_wanted() != self._format
                                      or self.settings.record.duration != self._duration):
                self._leave_look_back()     # off, or emptied and made again at the new size or duration
            elif self._stop_requested.is_set():
                self._hold()        # in look back too: the last seconds
        elif self._state == ScreenRecorderState.HELD:
            if self._discard_requested.is_set():
                self._discard()
            elif self._save_requested.is_set():
                self._save_held()
        elif self._state == ScreenRecorderState.IDLE and not self._busy():
            if self.settings.record.look_back:
                self._begin(timestamp, look_back=True)
            elif self._start_requested.is_set():
                self._begin(timestamp, look_back=False)
        elif self._state == ScreenRecorderState.ERROR and not self._busy() and self._start_requested.is_set():
            self._begin(timestamp, look_back=self.settings.record.look_back)      # after a failure only Start tries again

        self._capture_now = False
        if self._state == ScreenRecorderState.RECORDING:
            self._ticks += 1
            with self._lock:
                accepting = self._accepting
            self._capture_now = accepting and self._ticks % self._frame_step == 0

        self._start_requested.clear()
        self._stop_requested.clear()
        self._save_requested.clear()
        self._discard_requested.clear()
        self._play(timestamp)
        self._show(timestamp)

    def _measure(self, timestamp: float) -> None:
        if self._last_timestamp is not None:
            interval = timestamp - self._last_timestamp
            if interval > 0.0:
                self._fps += FPS_SMOOTH * (1.0 / interval - self._fps)
        self._last_timestamp = timestamp

    def _format_wanted(self) -> ScreenFormat:
        return ScreenFormat(even(self.settings.frame.width), even(self.settings.frame.height), self.settings.quality.chroma,
                            self.settings.quality.bit_depth)

    def _busy(self) -> bool:
        with self._lock:
            return self._take is not None or self._free or bool(self._queue)

    def _begin(self, timestamp: float, look_back: bool) -> None:
        """Start filling an empty store in the format set now; the capture is made in it (`capture_format`), the thread
        adds the blocks."""
        self._format = self._format_wanted()
        self._error_logged = False
        self._duration = self.settings.record.duration
        self._frame_step = self.settings.quality.frame_step
        self._ticks = 0
        maximum = self.settings.quality.max_duration
        limit = self._duration if self._duration > 0.0 and (maximum <= 0.0 or self._duration <= maximum) else 0.0
        target = frames_for(limit * LOOK_BACK_MARGIN, self._fps, self._frame_step) if look_back and limit else 0
        self._store.clear(self._format.shape, self._format.dtype)
        with self._lock:
            self._take_id += 1
            self._take_start = timestamp
            self._look_back = look_back
            self._limit = limit
            self._target = target
            self._written = 0
            self._stamps = []
            self._first = None
            self._complete = False
            self._reached = False
            self._dropped = 0
            self._accepting = True
        self._set_state(ScreenRecorderState.RECORDING)
        logger.info('Screen recording %s at %dx%d, %s %d-bit', 'look back' if look_back else 'started',
                    self._format.width, self._format.height, self._format.chroma.name, self._format.bit_depth)

    def _hold(self) -> None:
        """Hold the frames in the store, oldest first (in look back those of the last `duration` seconds), for in and
        out to be set on them; frames still on their way are let go, and a store without frames is freed."""
        with self._lock:
            self._accepting = False
            written, stamps = self._written, list(self._stamps)
            writing, look_back, limit = self._writing, self._look_back, self._limit
        slots = self._store.num_slots
        held = min(written, slots)
        first = written - held
        order = [(first + n) % slots for n in range(held)] if look_back else list(range(held))
        order = [slot for slot in order if slot != writing]     # a slot being written over is not the held frame
        if not order:
            logger.warning('Screen recording has no frames, nothing held')
            self._let_go()
            return
        held_stamps = [stamps[slot] for slot in order]
        if look_back and limit:
            newest = held_stamps[-1]
            kept = [n for n, t in enumerate(held_stamps) if newest - t <= limit]
            order, held_stamps = [order[n] for n in kept], [held_stamps[n] for n in kept]
        self._held_slots, self._held_stamps = order, held_stamps
        self._held_saved = False
        self._position = 0.0
        self.settings.take.in_out = [0.0, 1.0]
        self.settings.take.playhead = [0.0, self._held_length()]
        self._set_state(ScreenRecorderState.HELD)
        logger.info('Screen recording held: %d frames, %.2f s', len(order), self._held_length())

    def _play(self, timestamp: float) -> None:
        """Play the held take at its recorded times, looping between in and out; the dragged marker shows its own
        frame and play goes on from there. Only a held take plays; the switch goes off otherwise, a save included."""
        if self.settings.take.play and self._state is not ScreenRecorderState.HELD:
            self.settings.take.play = False
        self._playing = self.settings.take.play and bool(self._held_stamps)
        if not self._playing or self._dragging:
            self._play_from = None
            return
        start, length = self._held_stamps[0], self._held_length()
        mark_in, mark_out = sorted(self.settings.take.in_out)
        low, high = start + mark_in * length, start + mark_out * length
        if self._play_from is None:
            moment = start + self._position * length
            self._play_from = (timestamp, moment if low <= moment < high else low)
        began, moment = self._play_from
        span = high - low
        moment = low + ((moment - low) + (timestamp - began)) % span if span > 0.0 else low
        self._position = (moment - start) / length if length > 0.0 else 0.0

    def _held_length(self) -> float:
        """The held take's length by its frames' times (s)."""
        return self._held_stamps[-1] - self._held_stamps[0] if len(self._held_stamps) > 1 else 0.0

    def _save_held(self) -> None:
        """Write the held frames whose times lie between in and out; the take stays held."""
        start = self._held_stamps[0]
        length = self._held_length()
        mark_in, mark_out = sorted(self.settings.take.in_out)
        low, high = start + mark_in * length, start + mark_out * length
        kept = [n for n, t in enumerate(self._held_stamps) if low - 1e-9 <= t <= high + 1e-9]
        if not kept:
            logger.warning('Screen recording: no frames between in and out, nothing saved')
            return
        self._saving_held = True
        self._hand_over([self._held_slots[n] for n in kept], [self._held_stamps[n] for n in kept], free_after=False)

    def _discard(self) -> None:
        """Let the held take go: a look-back ring is emptied and records on, any other store is freed."""
        self._held_slots, self._held_stamps = [], []
        self.settings.take.playhead = [0.0, 0.0]
        if self._look_back and self.settings.record.look_back:
            self._ticks = 0
            with self._lock:
                self._take_id += 1
                self._written = 0
                self._stamps = []
                self._first = None
                self._dropped = 0
                self._accepting = True
            self._set_state(ScreenRecorderState.RECORDING)
            return
        self._let_go()

    def _let_go(self) -> None:
        """Have the thread free the store; the take ends there (`_finish`)."""
        with self._wake:
            self._accepting = False
            self._free = True
            self._wake.notify()
        self._set_state(ScreenRecorderState.ENCODING)

    def _hand_over(self, slots: list[int], stamps: list[float], free_after: bool) -> None:
        """Hand frames of the store, in their order, to the thread to encode into a new file."""
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = f'{stamp}_{self.settings.take.name}' if self.settings.take.name else stamp
        path = self._folder / f'{name}.mp4'
        part = 1
        while path.exists() or str(path) in self._paths:        # a second part saved within the same second
            part += 1
            path = self._folder / f'{name}_{part}.mp4'
        self._paths.add(str(path))
        with self._wake:
            self._accepting = False
            self._take = _Take(slots, stamps, str(path), self._format, self._dropped, free_after)
            self._wake.notify()
        self._set_state(ScreenRecorderState.ENCODING)

    def _leave_look_back(self) -> None:
        """Stop the look-back ring and have it freed; a new one starts after, if the switch is on."""
        self._let_go()

    def _finish(self) -> None:
        """The thread is done with a take or a free: a saved part of a held take returns to it, else the take ends."""
        self._done.clear()
        with self._lock:
            failed, self._failed = self._failed, False
        saving_held, self._saving_held = self._saving_held, False
        if failed:
            self._fail('the encode failed')
        elif saving_held:
            self._held_saved = True
            self._set_state(ScreenRecorderState.HELD)
            return
        if self._state == ScreenRecorderState.ERROR:
            return
        self._set_state(ScreenRecorderState.IDLE)

    def _fail(self, message: str) -> None:
        if not self._error_logged:
            logger.error('Screen recorder: %s', message)
            self._error_logged = True
        with self._wake:
            self._accepting = False
            self._free = True
            self._wake.notify()
        self._capture_now = False
        self._set_state(ScreenRecorderState.ERROR)

    def _set_state(self, state: ScreenRecorderState) -> None:
        self._state = state
        self.settings.record.state = state

    def _show(self, timestamp: float) -> None:
        if timestamp - self._shown_at < SHOW_INTERVAL:
            return
        self._shown_at = timestamp
        with self._lock:
            dropped = self._dropped
            first = self._first
            last = max(self._stamps) if self._stamps else None
        self.settings.record.dropped = dropped
        if self._state == ScreenRecorderState.HELD:
            length = self._held_length()
            self.settings.record.recorded = round(length, 2)
            self.settings.take.playhead = [round(self._position * length, 3), length]
            return
        self.settings.record.recorded = round(last - first, 2) if first is not None and last is not None else 0.0

    # THREAD: THE RING AND THE ENCODE, NO GL

    def _run(self) -> None:
        estimated_at = float('-inf')
        while True:
            with self._wake:
                if not (self._queue or self._take is not None or self._free or self._exit.is_set()):
                    self._wake.wait(POLL)
                if self._exit.is_set() and self._take is None and not self._free:
                    return
                frames = list(self._queue)
                self._queue.clear()
                take, free = self._take, self._free
            for frame in frames:
                self._store_frame(frame)
            if take is not None:
                self._write(take)
            elif free:
                self._release()
            now = time.perf_counter()
            if now - estimated_at >= ESTIMATE_INTERVAL:
                estimated_at = now
                self._estimate()

    def _release_queue(self) -> None:
        with self._lock:
            frames = list(self._queue)
            self._queue.clear()
        for frame in frames:
            frame.release()

    def _store_frame(self, frame: ScreenFrame) -> None:
        """Copy a frame into its slot in the store, growing the store ahead, and give the capture's buffer back."""
        try:
            with self._lock:
                if not self._accepting or frame.timestamp < self._take_start:
                    return
                take_id, look_back = self._take_id, self._look_back
                written, target = self._written, self._target
                self._dropped += frame.skipped
            if frame.data.shape != self._store.shape:
                return
            self._grow(written, target)
            slots = self._store.num_slots
            with self._lock:
                complete = self._complete
                if look_back and complete and slots:
                    slot = written % slots          # full: the newest frame replaces the oldest
                elif written < slots:
                    slot = written
                else:
                    if not look_back:
                        self._reached = True        # the memory is reached
                    return
                self._writing = slot
            np.copyto(self._store.frame(slot), frame.data)
            with self._lock:
                self._writing = None
                if not self._accepting or self._take_id != take_id:
                    return
                if slot < len(self._stamps):
                    self._stamps[slot] = frame.timestamp
                else:
                    self._stamps.append(frame.timestamp)
                self._written = written + 1
                if self._first is None:
                    self._first = frame.timestamp
                if not look_back and self._limit and frame.timestamp - self._first >= self._limit:
                    self._accepting = False     # the take spans `duration` by its frames' times
                    self._reached = True
        finally:
            frame.release()

    def _grow(self, written: int, target: int) -> None:
        """Add the next one-second block before the last one is full, while the free RAM stays above `ram_reserve`."""
        with self._lock:
            if self._complete:
                return
        slots = self._store.num_slots
        block = max(1, round(self._fps / max(1, self._frame_step)))
        if slots and written < slots - block * BLOCK_AHEAD:
            return
        if target:
            block = min(block, target - slots)
            if block <= 0:
                with self._lock:
                    self._complete = True
                return
        try:
            if self._free_memory() - block * self._store.frame_bytes < int(self.settings.record.ram_reserve * GB):
                logger.info('Screen recording store full at %d frames: free RAM reached the reserve', slots)
                with self._lock:
                    self._complete = True
                return
            self._store.add_block(block)
        except (MemoryError, ValueError) as e:
            logger.warning('Screen recording store stops at %d frames: %s', slots, e)
            with self._lock:
                self._complete = True

    def _estimate(self) -> None:
        """`max_duration`: what the free RAM, with the store's own, holds above `ram_reserve`."""
        frame_bytes = self._format_wanted().frame_bytes
        try:
            free = self._free_memory()
        except Exception as e:
            logger.debug('Screen recorder could not read the free RAM: %s', e)
            return
        usable = max(0, free + self._store.nbytes - int(self.settings.record.ram_reserve * GB))
        rate = frame_bytes * self._fps / max(1, self.settings.quality.frame_step)
        self.settings.quality.max_duration = round(usable / rate, 1)

    def _write(self, take: _Take) -> None:
        """Encode the take at the rate it was recorded at, then free the store when the take says so."""
        started = time.perf_counter()
        count = len(take.slots)
        span = take.stamps[-1] - take.stamps[0] if count > 1 else 0.0
        fps = (count - 1) / span if span > 0.0 else self._fps / max(1, self._frame_step)
        gaps = np.diff(np.asarray(take.stamps)) if count > 1 else np.zeros(1)
        comment = (f'frames {count}, fps {fps:.3f}, mean gap {gaps.mean() * 1000:.2f} ms, '
                   f'worst gap {gaps.max() * 1000:.2f} ms, dropped {take.dropped}')
        ok = False
        try:
            Path(take.path).parent.mkdir(parents=True, exist_ok=True)
            sink = self._clip_sink(take.path, fps, self.settings.take.quality, take.format)
            sink.comment = comment
            for slot in take.slots:
                sink.add(self._store.frame(slot))
            ok = sink.close()
        except Exception as e:
            logger.error('Screen recording encode of %s failed: %s', take.path, e)
        seconds = time.perf_counter() - started
        if ok:
            logger.info('Screen recording %s: %s, encoded in %.1f s', take.path, comment, seconds)
        else:
            logger.error('Screen recording %s not written', take.path)
        if take.free_after:
            self._free_store()
        with self._lock:
            self._take = None
            self._failed = not ok
        self._done.set()

    def _release(self) -> None:
        self._free_store()
        with self._lock:
            self._free = False
        self._done.set()

    def _free_store(self) -> None:
        self._store.clear()
        with self._lock:
            self._written = 0
            self._stamps = []
            self._first = None
            self._complete = False
            self._reached = False
