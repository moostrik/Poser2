"""ScreenRecorder: records the app's screen frames into a ring in video memory and encodes it to an HEVC mp4.

The render tick (`update`) copies one screen frame per tick, or per second tick, into the ring that a
`ScreenRing` (the GL package's `ScreenCapture`) holds; the recorder's own thread grows the ring by one-second
blocks while it records and encodes it after Stop or Save, then frees it. Buttons only set Events; the render
tick acts on them. Design and reasons: modules/video/VIDEO.md.
"""

import logging
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Protocol

import numpy as np

from modules.settings import BaseSettings, Field, Widget

from .ffmpeg import ClipCodec, ClipInput, ClipWriter, Encoder

logger = logging.getLogger(__name__)

GB: int = 1 << 30
DEFAULT_FPS: float = 60.0       # ticks/s assumed until the ticks are measured
FPS_SMOOTH: float = 0.05        # share of a new tick interval in the measured rate
BLOCK_AHEAD: float = 0.5        # share of a block left when the next one is added
LOOK_BACK_MARGIN: float = 1.2   # the look-back ring's frames over `duration` at the rate measured, for faster ticks
POLL: float = 0.05              # s between the thread's looks at the ring while it records
ESTIMATE_INTERVAL: float = 1.0  # s between estimates of `max_duration`
SHOW_INTERVAL: float = 0.25     # s between updates of `recorded`


class ScreenSize(Enum):
    """A recording size preset, landscape (width, height); portrait swaps them."""
    HD =     (1920, 1080)
    QHD =    (2560, 1440)
    UHD =    (3840, 2160)
    CUSTOM = (0, 0)


class ScreenMemory(Enum):
    """Where the ring of recorded frames lives."""
    HOST =  auto()      # pinned RAM: long takes
    VIDEO = auto()      # video memory: short takes, nothing over PCIe while recording


class ScreenRecorderState(Enum):
    IDLE =      auto()
    RECORDING = auto()
    ENCODING =  auto()
    ERROR =     auto()


class ScreenRecorderSettings(BaseSettings):
    start:        Field[bool]                = Field(False, widget=Widget.button, description="Start screen recording")
    stop:         Field[bool]                = Field(False, widget=Widget.button, description="Stop screen recording")
    look_back:    Field[bool]                = Field(False, widget=Widget.switch, description="Keep the last seconds all the time")
    save:         Field[bool]                = Field(False, widget=Widget.button, description="Save the last seconds")
    name:         Field[str]                 = Field("", widget=Widget.input, description="Added to the file name")
    main_window:  Field[bool]                = Field(False, widget=Widget.switch, description="Record the main window instead of the sources")
    fill:         Field[bool]                = Field(False, widget=Widget.switch, description="Fill the frame and cut the overhang, instead of black bars")
    size:         Field[ScreenSize]          = Field(ScreenSize.UHD, description="Recording size preset")
    portrait:     Field[bool]                = Field(True, widget=Widget.switch, description="Upright: width and height swapped")
    width:        Field[int]                 = Field(2160, min=2, max=8192, step=2, description="Recording width (px); a change sets the preset to CUSTOM")
    height:       Field[int]                 = Field(3840, min=2, max=8192, step=2, description="Recording height (px); a change sets the preset to CUSTOM")
    snap:         Field[bool]                = Field(False, widget=Widget.button, description="Set the size to the first source's")
    frame_step:   Field[int]                 = Field(1, min=1, max=2, step=1, description="Record every tick (1), or every other (2)")
    duration:     Field[float]               = Field(0.0, min=0.0, max=120.0, description="Recording length (s); 0 or above the maximum uses the maximum")
    memory:       Field[ScreenMemory]        = Field(ScreenMemory.HOST, description="Where the recorded frames are held: RAM or video memory")
    max_duration: Field[float]               = Field(0.0, access=Field.READ, description="Longest recording the free memory holds (s)")
    reserve:      Field[float]               = Field(1.0, min=0.0, max=8.0, description="Video memory left free for the rest of the app (GB)")
    ram_reserve:  Field[float]               = Field(8.0, min=1.0, max=64.0, description="RAM left free for Windows and the app (GB)")
    quality:      Field[int]                 = Field(18, min=0, max=51, description="Constant quality, lower is better")
    state:        Field[ScreenRecorderState] = Field(ScreenRecorderState.IDLE, access=Field.READ, description="Idle, recording, encoding")
    recorded:     Field[float]               = Field(0.0, access=Field.READ, description="Seconds in the ring (s)")


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


class ScreenRing(Protocol):
    """What the recorder needs from the screen capture: the screen frame's copy into a ring of NV12 frames.
    `allocate`, `deallocate`, `capture` and `flush` run on the render thread, the rest on any thread."""

    @property
    def allocated(self) -> bool: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def frame_bytes(self) -> int: ...
    @property
    def num_slots(self) -> int: ...
    @property
    def host_memory(self) -> bool: ...
    def allocate(self, width: int, height: int, host_memory: bool) -> None: ...
    def deallocate(self) -> None: ...
    def capture(self, slot: int) -> None: ...
    def flush(self) -> None: ...
    def free_memory(self, host_memory: bool) -> int: ...
    def add_block(self, slots: int) -> None: ...
    def free_blocks(self) -> None: ...
    def synchronize(self) -> None: ...
    def queue_download(self, slot: int) -> None: ...
    def wait_download(self) -> np.ndarray: ...


class ClipSink(Protocol):
    """What the encode writes NV12 frames to."""
    comment: str
    def add(self, image: np.ndarray) -> None: ...
    def close(self) -> bool: ...


ClipSinkFactory = Callable[[str, float, int], ClipSink]     # (path, fps, quality)


def hevc_clip(path: str, fps: float, quality: int) -> ClipSink:
    """An HEVC mp4 on NVENC from NV12 frames."""
    return ClipWriter(path, fps, Encoder.NVENC, ClipInput.NV12, ClipCodec.HEVC, quality)


class _Take:
    """One encode: the ring's slots in recorded order, their tick times, the file, and whether the ring is freed
    after it."""

    def __init__(self, slots: list[int], stamps: list[float], path: str, free_after: bool) -> None:
        self.slots: list[int] = slots
        self.stamps: list[float] = stamps
        self.path: str = path
        self.free_after: bool = free_after


class ScreenRecorder:
    """Screen recording of the frames drawn into the ring's screen frame: Start / Stop, or look back with Save."""

    def __init__(self, settings: ScreenRecorderSettings, ring: ScreenRing, folder: str,
                 clip_sink: ClipSinkFactory = hevc_clip) -> None:
        self.settings: ScreenRecorderSettings = settings
        self._ring: ScreenRing = ring
        self._folder: Path = Path(folder)
        self._clip_sink: ClipSinkFactory = clip_sink

        self._start_requested: Event = Event()
        self._stop_requested: Event = Event()
        self._save_requested: Event = Event()
        self._exit: Event = Event()

        # shared with the thread, under the lock
        self._lock: Lock = Lock()
        self._recording: bool = False       # the ring is filled on the render tick
        self._target: int = 0               # slots the ring grows to; 0: as many as the memory holds
        self._written: int = 0              # frames captured into the ring since it was emptied
        self._complete: bool = False        # the ring grows no more
        self._take: _Take | None = None     # handed to the thread to encode
        self._free: bool = False            # the thread is to free the ring without an encode
        self._failed: bool = False          # the last take was not written
        self._done: Event = Event()         # the thread finished a take or a free

        # render thread only
        self._state: ScreenRecorderState = ScreenRecorderState.IDLE
        self._look_back: bool = False       # the ring in use is a look-back ring
        self._size: tuple[int, int] = (0, 0)
        self._duration: float = 0.0         # the duration the look-back ring was made for
        self._limit: float = 0.0            # s a take spans, by its frames' times; 0: until the memory stops it
        self._frame_step: int = 1
        self._ticks: int = 0
        self._stamps: list[float] = []      # tick time of each slot's frame
        self._pending: bool = False         # a frame waits in the ring's other pixel buffer
        self._fps: float = DEFAULT_FPS
        self._last_timestamp: float | None = None
        self._shown_at: float = 0.0
        self._error_logged: bool = False

        self._thread: Thread = Thread(target=self._run, name='screen recorder', daemon=True)

        self.settings.bind(ScreenRecorderSettings.start, self._on_start)
        self.settings.bind(ScreenRecorderSettings.stop, self._on_stop)
        self.settings.bind(ScreenRecorderSettings.save, self._on_save)
        self.settings.bind(ScreenRecorderSettings.size, self._on_size)
        self.settings.bind(ScreenRecorderSettings.portrait, self._on_portrait)
        self.settings.bind(ScreenRecorderSettings.width, self._on_width_height)
        self.settings.bind(ScreenRecorderSettings.height, self._on_width_height)

    @property
    def capturing(self) -> bool:
        """Whether the screen frame is to be drawn this tick: the next `update` copies it into the ring."""
        with self._lock:
            return self._recording and (self._ticks + 1) % self._frame_step == 0

    @property
    def fps(self) -> float:
        """The render ticks per second, measured."""
        return self._fps

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the recording; a take in progress is encoded up to its last copied frame, then the ring's CUDA
        memory is freed. The render thread frees the GL objects in `deallocate`."""
        self.settings.unbind(ScreenRecorderSettings.start, self._on_start)
        self.settings.unbind(ScreenRecorderSettings.stop, self._on_stop)
        self.settings.unbind(ScreenRecorderSettings.save, self._on_save)
        self.settings.unbind(ScreenRecorderSettings.size, self._on_size)
        self.settings.unbind(ScreenRecorderSettings.portrait, self._on_portrait)
        self.settings.unbind(ScreenRecorderSettings.width, self._on_width_height)
        self.settings.unbind(ScreenRecorderSettings.height, self._on_width_height)
        if self._state == ScreenRecorderState.RECORDING and not self._look_back:
            copied = len(self._stamps) - (1 if self._pending else 0)
            self._encode(copied, free_after=True)
        self._exit.set()
        if self._thread.is_alive():
            self._thread.join()
        try:
            self._ring.free_blocks()
        except Exception as e:
            logger.warning('Screen recorder could not free its ring: %s', e)

    def deallocate(self) -> None:
        """Free the ring's GL objects; render thread, after `stop`."""
        if self._ring.allocated:
            self._ring.deallocate()

    # BUTTONS AND SIZE RULES (any thread)

    def _on_start(self, _: bool) -> None:
        self._start_requested.set()

    def _on_stop(self, _: bool) -> None:
        self._stop_requested.set()

    def _on_save(self, _: bool) -> None:
        self._save_requested.set()

    def _on_size(self, size: ScreenSize) -> None:
        if size is ScreenSize.CUSTOM:
            return
        width, height = preset_size(size, self.settings.portrait)
        self.settings.width = width
        self.settings.height = height

    def _on_portrait(self, portrait: bool) -> None:
        width, height = self.settings.width, self.settings.height
        if portrait != (height > width) and width != height:
            self.settings.width = height
            self.settings.height = width

    def _on_width_height(self, _: int) -> None:
        width, height = self.settings.width, self.settings.height
        self.settings.size = matching_preset(width, height)
        if width != height:
            self.settings.portrait = height > width

    # RENDER TICK

    def update(self, timestamp: float) -> None:
        """Copy this tick's screen frame when `capturing` said so, then act on the buttons and the thread's news."""
        self._measure(timestamp)
        if self._done.is_set():
            self._finish()

        if self._state == ScreenRecorderState.RECORDING:
            self._record(timestamp)

        if self._state == ScreenRecorderState.RECORDING:
            if self._look_back:
                if not self.settings.look_back:
                    self._leave_look_back()
                elif (self._size_wanted() != self._size or self.settings.duration != self._duration
                      or (self.settings.memory is ScreenMemory.HOST) != self._ring.host_memory):
                    self._leave_look_back()     # emptied, and made again at the new size or duration
                elif self._save_requested.is_set():
                    self._save_requested.clear()
                    self._encode(self._flush(), free_after=False)
            elif self._stop_requested.is_set():
                self._encode(self._flush(), free_after=True)
        elif self._state == ScreenRecorderState.IDLE and not self._busy():
            if self.settings.look_back:
                self._begin(look_back=True)
            elif self._start_requested.is_set():
                self._begin(look_back=False)
        elif self._state == ScreenRecorderState.ERROR and not self._busy() and self._start_requested.is_set():
            self._begin(look_back=self.settings.look_back)      # after a failure only Start tries again

        self._start_requested.clear()
        self._stop_requested.clear()
        if not self._look_back or self._state != ScreenRecorderState.RECORDING:
            self._save_requested.clear()
        self._show(timestamp)

    def _measure(self, timestamp: float) -> None:
        if self._last_timestamp is not None:
            interval = timestamp - self._last_timestamp
            if interval > 0.0:
                self._fps += FPS_SMOOTH * (1.0 / interval - self._fps)
        self._last_timestamp = timestamp

    def _size_wanted(self) -> tuple[int, int]:
        return even(self.settings.width), even(self.settings.height)

    def _busy(self) -> bool:
        with self._lock:
            return self._take is not None or self._free

    def _begin(self, look_back: bool) -> None:
        """Make the ring's GL objects at the size set now and start filling it; the thread adds its blocks."""
        size = self._size_wanted()
        host_memory = self.settings.memory is ScreenMemory.HOST
        try:
            if (not self._ring.allocated or (self._ring.width, self._ring.height) != size
                    or self._ring.host_memory != host_memory):
                self._ring.allocate(*size, host_memory)
        except Exception as e:
            self._fail(f'could not allocate the screen capture at {size[0]}x{size[1]}: {e}')
            return
        self._error_logged = False
        self._size = size
        self._look_back = look_back
        self._duration = self.settings.duration
        self._frame_step = self.settings.frame_step
        self._ticks = 0
        self._stamps = []
        self._pending = False
        maximum = self.settings.max_duration
        self._limit = self._duration if self._duration > 0.0 and (maximum <= 0.0 or self._duration <= maximum) else 0.0
        target = 0      # 0: the ring grows as long as the memory holds
        if look_back and self._limit:
            target = frames_for(self._limit * LOOK_BACK_MARGIN, self._fps, self._frame_step)
        with self._lock:
            self._target = target
            self._written = 0
            self._complete = False
            self._recording = True
        self._set_state(ScreenRecorderState.RECORDING)
        logger.info('Screen recording %s at %dx%d', 'look back' if look_back else 'started', *size)

    def _record(self, timestamp: float) -> None:
        self._ticks += 1
        if self._ticks % self._frame_step:
            return
        with self._lock:
            slots = self._ring.num_slots
            written, complete, target = self._written, self._complete, self._target
        if self._look_back and complete and slots > 0:
            slot = written % slots      # full: the newest frame replaces the oldest
        elif written < slots:
            slot = written
        elif complete or (target and written >= target):
            if not self._look_back:
                self._encode(self._flush(), free_after=True)    # the memory is reached
            return
        else:
            return      # the next block is not there yet: this frame is left out
        try:
            self._ring.capture(slot)
        except Exception as e:
            self._fail(f'screen capture failed: {e}')
            return
        if slot < len(self._stamps):
            self._stamps[slot] = timestamp
        else:
            self._stamps.append(timestamp)
        self._pending = True
        with self._lock:
            self._written = written + 1
        if not self._look_back and self._limit and timestamp - self._stamps[0] >= self._limit:
            self._encode(self._flush(), free_after=True)    # the take spans `duration` by its frames' times

    def _flush(self) -> int:
        """Copy the pending frame into the ring and stop filling it; the number of frames recorded."""
        with self._lock:
            self._recording = False
            written = self._written
        if self._pending:
            try:
                self._ring.flush()
            except Exception as e:
                logger.error('Screen capture flush failed: %s', e)
                written -= 1
            self._pending = False
        return written

    def _encode(self, written: int, free_after: bool) -> None:
        """Hand the frames in the ring, oldest first, to the thread to encode."""
        with self._lock:
            slots = self._ring.num_slots
        held = min(written, slots)
        if held <= 0:
            logger.warning('Screen recording has no frames, nothing saved')
            if free_after:
                with self._lock:
                    self._free = True
                self._set_state(ScreenRecorderState.ENCODING)
            return
        first = written - held
        order = [(first + n) % slots for n in range(held)] if self._look_back else list(range(held))
        stamps = [self._stamps[slot] for slot in order]
        if self._look_back and self._limit:
            # the frames of the last `duration` seconds, by their times
            newest = stamps[-1]
            kept = [n for n, t in enumerate(stamps) if newest - t <= self._limit]
            order, stamps = [order[n] for n in kept], [stamps[n] for n in kept]
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = f'{stamp}_{self.settings.name}' if self.settings.name else stamp
        take = _Take(order, stamps, str(self._folder / f'{name}.mp4'), free_after)
        with self._lock:
            self._recording = False
            self._take = take
        self._set_state(ScreenRecorderState.ENCODING)

    def _leave_look_back(self) -> None:
        """Stop the look-back ring and have it freed; a new one starts after, if the switch is on."""
        self._flush()
        with self._lock:
            self._free = True
        self._set_state(ScreenRecorderState.ENCODING)

    def _finish(self) -> None:
        """The thread is done with a take or a free: go on recording into a look-back ring that is kept, else free
        the GL objects."""
        self._done.clear()
        with self._lock:
            kept = self._ring.num_slots > 0
            failed, self._failed = self._failed, False
        if failed:
            self._fail('the encode failed')
        if self._state == ScreenRecorderState.ERROR:
            if self._ring.allocated and not kept:
                self._ring.deallocate()
            return
        if self._look_back and kept and self.settings.look_back:
            with self._lock:
                self._recording = True
            self._set_state(ScreenRecorderState.RECORDING)
            return
        self._look_back = False
        if self._ring.allocated:
            self._ring.deallocate()
        self._set_state(ScreenRecorderState.IDLE)

    def _fail(self, message: str) -> None:
        if not self._error_logged:
            logger.error('Screen recorder: %s', message)
            self._error_logged = True
        with self._lock:
            self._recording = False
            self._free = True
        self._pending = False
        self._set_state(ScreenRecorderState.ERROR)

    def _set_state(self, state: ScreenRecorderState) -> None:
        self._state = state
        self.settings.state = state

    def _show(self, timestamp: float) -> None:
        if timestamp - self._shown_at < SHOW_INTERVAL:
            return
        self._shown_at = timestamp
        with self._lock:
            held = min(self._written, self._ring.num_slots)
        self.settings.recorded = round(held * self._frame_step / max(self._fps, 1.0), 2)

    # THREAD: CUDA MEMORY AND THE ENCODE, NO GL

    def _run(self) -> None:
        estimated_at = float('-inf')
        while not self._exit.is_set() or self._has_work():
            now = time.perf_counter()
            if now - estimated_at >= ESTIMATE_INTERVAL:
                estimated_at = now
                self._estimate()
            with self._lock:
                take, free = self._take, self._free
            if take is not None:
                self._write(take)
            elif free:
                self._release()
            else:
                self._grow()
            if not self._has_work():
                self._exit.wait(POLL)

    def _has_work(self) -> bool:
        with self._lock:
            return self._take is not None or self._free

    def _reserve(self, host_memory: bool) -> int:
        """Bytes left free in the ring's memory: `ram_reserve` in RAM, `reserve` in video memory."""
        return int((self.settings.ram_reserve if host_memory else self.settings.reserve) * GB)

    def _estimate(self) -> None:
        """`max_duration`: what the free memory set in `memory`, with the ring's own there, holds above its reserve."""
        width, height = even(self.settings.width), even(self.settings.height)
        frame_bytes = width * height * 3 // 2
        host_memory = self.settings.memory is ScreenMemory.HOST
        try:
            free = self._ring.free_memory(host_memory)
        except Exception as e:
            logger.debug('Screen recorder could not read the free memory: %s', e)
            return
        with self._lock:
            slots = self._ring.num_slots
            held = slots * self._ring.frame_bytes if slots and self._ring.host_memory == host_memory else 0
        usable = max(0, free + held - self._reserve(host_memory))
        rate = frame_bytes * self._fps / max(1, self.settings.frame_step)
        self.settings.max_duration = round(usable / rate, 1)

    def _grow(self) -> None:
        """Add the next one-second block before the last one is full, while the free memory stays above reserve."""
        with self._lock:
            if not self._recording or self._complete:
                return
            written, target = self._written, self._target
        slots = self._ring.num_slots
        block = max(1, round(self._fps / max(1, self._frame_step)))
        if slots and written < slots - block * BLOCK_AHEAD:
            return
        if target:
            block = min(block, target - slots)
            if block <= 0:
                with self._lock:
                    self._complete = True
                return
        host_memory = self._ring.host_memory
        try:
            free = self._ring.free_memory(host_memory)
            if free - block * self._ring.frame_bytes < self._reserve(host_memory):
                logger.info('Screen recording ring full at %d frames: free %s reached the reserve', slots,
                            'RAM' if host_memory else 'video memory')
                with self._lock:
                    self._complete = True
                return
            self._ring.add_block(block)
        except Exception as e:
            logger.warning('Screen recording ring stops at %d frames: %s', slots, e)
            with self._lock:
                self._complete = True

    def _write(self, take: _Take) -> None:
        """Encode the take at the rate it was recorded at, then free the ring when the take says so."""
        started = time.perf_counter()
        count = len(take.slots)
        span = take.stamps[-1] - take.stamps[0] if count > 1 else 0.0
        fps = (count - 1) / span if span > 0.0 else self._fps / max(1, self._frame_step)
        gaps = np.diff(np.asarray(take.stamps)) if count > 1 else np.zeros(1)
        comment = f'frames {count}, fps {fps:.3f}, mean gap {gaps.mean() * 1000:.2f} ms, worst gap {gaps.max() * 1000:.2f} ms'
        ok = False
        try:
            take_folder = Path(take.path).parent
            take_folder.mkdir(parents=True, exist_ok=True)
            self._ring.synchronize()
            sink = self._clip_sink(take.path, fps, self.settings.quality)
            sink.comment = comment
            self._ring.queue_download(take.slots[0])
            for n in range(count):
                if n + 1 < count:
                    self._ring.queue_download(take.slots[n + 1])
                sink.add(self._ring.wait_download())
            ok = sink.close()
        except Exception as e:
            logger.error('Screen recording encode of %s failed: %s', take.path, e)
        seconds = time.perf_counter() - started
        if ok:
            logger.info('Screen recording %s: %s, encoded in %.1f s', take.path, comment, seconds)
        else:
            logger.error('Screen recording %s not written', take.path)
        if take.free_after:
            self._free_ring()
        with self._lock:
            self._take = None
            self._failed = not ok
        self._done.set()

    def _release(self) -> None:
        self._free_ring()
        with self._lock:
            self._free = False
        self._done.set()

    def _free_ring(self) -> None:
        try:
            self._ring.free_blocks()
        except Exception as e:
            logger.error('Screen recorder could not free its ring: %s', e)
        with self._lock:
            self._written = 0
            self._complete = False
