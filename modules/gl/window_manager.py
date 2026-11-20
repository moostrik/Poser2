from __future__ import annotations

import logging
from threading import Event, Lock, Thread, current_thread
from time import perf_counter
from typing import Callable, TYPE_CHECKING

from OpenGL.GL import glBindFramebuffer, glClientWaitSync, glDeleteSync, glFenceSync, glFlush, glGetString, glViewport, \
    GL_FRAMEBUFFER, GL_SYNC_FLUSH_COMMANDS_BIT, GL_SYNC_GPU_COMMANDS_COMPLETE, GL_VERSION
import glfw

if TYPE_CHECKING:
    from .render_base import RenderBase
from modules.utils import FPS, PerformanceTimer, Waker
from .monitor import Monitor
from .window import Window
from .window_input import WindowInput
from .window_settings import WindowManagerSettings, WindowSettings

logger = logging.getLogger(__name__)


class WindowManager:
    """The window loop on the main thread: GLFW's windows, monitors and events, the renderer's update once a frame
    and its draw in each shown window."""
    _MIN_VISIBLE_SHARE = 0.5    # share of a window that must be on the monitors at start, else the default placement
    # The default placement: monitor 0, relative to it
    _DEFAULT_X = 20
    _DEFAULT_Y = 40
    _DEFAULT_WIDTH = 1280
    _DEFAULT_HEIGHT = 720

    def __init__(self, renderer: RenderBase, settings: WindowManagerSettings) -> None:
        """`settings` an app's `windows` group: the loop's fields and its declared windows."""
        self.renderer = renderer
        self.settings: WindowManagerSettings = settings
        self.window_settings: list[WindowSettings] = settings.windows      # fixed at start: every window is made
        self._window_names: list[str] = settings.window_names
        self.input = WindowInput()
        self.input.add_key_callback(self._on_key)

        self._fps = FPS()
        # CPU time of the renderer's calls; the GPU work they queue is not waited for
        self._update_timer = PerformanceTimer(name="Render update", sample_count=200)
        self._draw_timer = PerformanceTimer(name="Render draw", sample_count=200)
        self._interval_timer = PerformanceTimer(name="Render interval", sample_count=200)
        self._last_update: float = 0.0
        self.frame_time: float = 0.0        # the current frame's stamp (perf_counter), set before its update

        self._loop_thread: Thread | None = None     # the thread that runs the loop, set by run()
        self._close_request = Event()               # stop(), Esc or a closed window asked the loop to end
        self._finished = Event()                    # the loop has ended and cleaned up
        self._finished.set()                        # no loop yet
        self._window_lock = Lock()                  # the windows' handles, for stop() from another thread

        self._windows: list[Window] = []
        self._sizes: list[tuple[int, int]] = []     # each window's framebuffer size, as the renderer knows it
        self._monitors: list[Monitor] = []          # the connected monitors, in MonitorId order
        self._monitors_changed: bool = False        # a hot-plug, handled at the top of the next frame
        self._monitor_cb = None                     # strong ref to the monitor callback (prevents GC)
        self._swap_intervals: tuple[bool, int] | None = None    # (v_sync, v_sync_window) as applied
        self._waker = Waker()                       # the frame pacing's wait, ended at once by stop() and a new limit
        self._next_frame: float = 0.0               # the paced frame's deadline (perf_counter), 0 when not paced
        self._swap_fences: list = []                # a fence after each paced swap, the newest last

        self._exit_callbacks: list[Callable[[], None]] = []
        self._callback_lock = Lock()

        self._place_default: bool = False           # the button, applied at the tick
        self._cb_default = lambda v: self._ask_default_placement() if v else None
        self.settings.bind(WindowManagerSettings.default_placement, self._cb_default)
        self._cb_fps_limit = lambda _v: self._waker.set()    # a new limit ends the wait at once: none outlasts it
        self.settings.bind(WindowManagerSettings.fps_limit, self._cb_fps_limit)

    # LIFECYCLE
    def run(self) -> None:
        """The window loop on the calling thread, the main thread (GLFW's rule), until the windows close."""
        self._loop_thread = current_thread()
        self._finished.clear()
        try:
            self._setup()
            self._render_loop()
        except Exception:
            logger.exception("Error in the window loop")
        finally:
            self._cleanup()
            self._finished.set()

    def stop(self) -> None:
        """Ask the window loop to end; from any thread. From another thread, wait up to 2 s for it to end."""
        self.input.clear_callbacks()
        self._close_request.set()
        with self._window_lock:                         # the loop's cleanup destroys the windows under it
            if self._windows:
                glfw.set_window_should_close(self._windows[0].handle, True)     # both allowed from any thread
                glfw.post_empty_event()
        self._waker.set()                               # a set() after the loop's close does nothing
        if self._loop_thread is None or current_thread() is self._loop_thread:
            return
        if not self._finished.wait(2.0):
            logger.warning("The window loop did not end within 2 s")

    def window_size(self, window: int) -> tuple[int, int]:
        """A window's framebuffer size (pixels), as the renderer knows it: from `allocate` on."""
        return self._sizes[window]

    def add_exit_callback(self, callback: Callable[[], None]) -> None:
        with self._callback_lock:
            if callback not in self._exit_callbacks:
                self._exit_callbacks.append(callback)

    def _setup(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._monitors = Monitor.connected()
        self._write_connected()
        if self.settings.v_sync_window not in self._window_names:
            logger.warning("v_sync_window %r is not one of the windows %s, using %s", self.settings.v_sync_window,
                           self._window_names, self._window_names[0])
            self.settings.v_sync_window = self._window_names[0]
        title: str = self.settings.title
        with self._window_lock:
            for index, settings in enumerate(self.window_settings):
                if self._off_screen(settings):
                    logger.warning("Window %s is under half on the monitors, placing it at the default place on its monitor", index)
                    self._write_default_placement(settings)
                share: Window | None = self._windows[0] if self._windows else None
                window = Window(index, settings, title if index == 0 else f"{title} {index}", share, self._close_request.set)
                self.input.attach(window)
                self._windows.append(window)

        self._windows[0].make_current()
        version = glGetString(GL_VERSION)
        if not isinstance(version, bytes):
            raise RuntimeError("OpenGL context is not valid")
        logger.info("OpenGL version: %s", version.decode("utf-8"))

        for window in self._windows:
            window.apply(window.placement(self._monitors, self.settings.fullscreen_mouse))
        for window in self._windows:            # in order, so a later window is in front of window 0
            if window.shown:
                glfw.focus_window(window.handle)
        self._apply_swap_intervals()
        self._sizes = [window.framebuffer_size() for window in self._windows]

        self._monitor_cb = self._on_monitor_change
        glfw.set_monitor_callback(self._monitor_cb)

    def _cleanup(self) -> None:
        self.settings.unbind_all(self._cb_default)
        self.settings.unbind_all(self._cb_fps_limit)
        try:
            glfw.set_monitor_callback(None)  # pyright: ignore[reportArgumentType]
            self._monitor_cb = None
            with self._window_lock:
                for window in reversed(self._windows):
                    window.destroy()
                self._windows.clear()
                glfw.terminate()
            self._waker.close()
        except Exception:
            logger.exception("Error cleaning up GLFW")
        with self._callback_lock:
            callbacks = list(self._exit_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Error in exit callback")

    # THE LOOP
    def _render_loop(self) -> None:
        main: Window = self._windows[0]
        main.make_current()            # the renderer's objects live in window 0's context
        self.renderer.allocate()
        try:
            while not self._close_request.is_set() and not any(glfw.window_should_close(w.handle) for w in self._windows):
                main.make_current()    # drawing a window leaves its context current
                self._tick_windows()
                self._update()
                self._draw()
                self._pace()
                try:
                    glfw.poll_events()
                except Exception:
                    # Raised by one of the callbacks (key, mouse, window, monitor): it must not end the loop
                    logger.exception("Error in event callback")
        finally:
            main.make_current()
            for fence in self._swap_fences:
                glDeleteSync(fence)
            self._swap_fences.clear()
            self.renderer.deallocate()

    def _tick_windows(self) -> None:
        """The monitors after a hot-plug, the user's moves written, the settings' differences applied, and the
        renderer told of each changed size; in window 0's context."""
        if self._monitors_changed:
            self._monitors_changed = False
            self._monitors = Monitor.connected()
            self._write_connected()
            logger.info("Monitors changed: %s", self.settings.connected)
        if self._place_default:
            self._place_default = False
            for settings in self.window_settings:
                self._write_default_placement(settings)
        mouse: bool = self.settings.fullscreen_mouse
        for window in self._windows:
            window.write_move(self._monitors, mouse)
            window.apply(window.placement(self._monitors, mouse))
        shown: list[str] = [self._window_names[w.index] for w in self._windows if w.shown]
        if shown != self.settings.shown_windows:
            self.settings.shown_windows = shown
        self._apply_swap_intervals()
        for index, window in enumerate(self._windows):
            size: tuple[int, int] = window.framebuffer_size()
            if size != self._sizes[index] and size[0] > 0 and size[1] > 0:
                self._sizes[index] = size
                try:
                    self.renderer.on_window_resize(index, *size)
                except Exception:
                    logger.exception("Error in on_window_resize")

    def _update(self) -> None:
        """Update renderer state. Called once per frame; ``frame_time`` is the frame's stamp."""
        self._fps.processed()
        self.settings.avg_fps = int(self._fps.get_rate_average())
        self.settings.min_fps = int(self._fps.get_rate_minimum())
        start = perf_counter()
        if self._last_update:
            self._interval_timer.add_time((start - self._last_update) * 1000.0, report=False)
        self._last_update = start
        self.frame_time = start
        try:
            self.renderer.update()
        except Exception:
            logger.exception("Error in update")
        self._update_timer.add_time((perf_counter() - start) * 1000.0, report=False)

    def _draw(self) -> None:
        """Each shown window drawn, the others swapped, then the v-sync window last."""
        shown: list[Window] = [w for w in self._windows if w.shown]
        self._hold_frame_in_flight()
        start = perf_counter()
        for window in shown:
            window.make_current()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            width, height = self._sizes[window.index]
            glViewport(0, 0, width, height)
            try:
                self.renderer.draw(window.index, width, height)
            except Exception:
                logger.exception("Error in draw of window %s", window.index)
        self._draw_timer.add_time((perf_counter() - start) * 1000.0, report=False)
        v_sync_index: int = self._v_sync_index()
        for window in shown:
            if window.index != v_sync_index:
                window.make_current()
                glfw.swap_buffers(window.handle)
        for window in shown:
            if window.index == v_sync_index:
                window.make_current()
                glfw.swap_buffers(window.handle)
                self._swap_fences.append(glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0))
                glFlush()                   # the fence sent to the GPU from this context, where it was made

    _FENCE_TIMEOUT_NS: int = 100_000_000    # a fence not passed in 100 ms is given up on

    def _hold_frame_in_flight(self) -> None:
        """Before a frame draws, a wait for the GPU to pass the paced swap before last: one frame stays in flight, so the
        driver queues no more than one (the fences are shared between the windows' contexts)."""
        while len(self._swap_fences) >= 2:
            fence = self._swap_fences.pop(0)
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, self._FENCE_TIMEOUT_NS)
            glDeleteSync(fence)

    def _pace(self) -> None:
        """Without v-sync pacing the loop, a wait until the next frame's deadline at `fps_limit`: an absolute deadline
        on perf_counter, a period on from the last; a frame that ends after its deadline goes on at once, and more than
        a period behind the deadline moves to now. Every window hidden and `fps_limit` 0: 60 fps."""
        s = self.settings
        shown: list[Window] = [w for w in self._windows if w.shown]
        v_sync_index: int = self._v_sync_index()
        v_synced: bool = s.v_sync and any(w.index == v_sync_index for w in shown)
        limit: float = s.fps_limit                          # a negative is off; a fraction of 1 fps is 1: a wait of 1 s at most
        rate: float = max(1.0, limit) if limit > 0.0 else (0.0 if shown else 60.0)
        if v_synced or rate <= 0.0:
            self._next_frame = 0.0
            return
        period: float = 1.0 / rate
        now: float = perf_counter()
        deadline: float = self._next_frame + period if self._next_frame else now + period
        if now - deadline > period:
            self._next_frame = now          # behind: no wait, and no run of short frames to catch up
            return
        self._next_frame = deadline
        if deadline > now:
            self._waker.wait(deadline)

    def _apply_swap_intervals(self) -> None:
        """V-sync on the v-sync window only, the others swapped without it; applied on a change."""
        wanted: tuple[bool, int] = (self.settings.v_sync, self._v_sync_index())
        if wanted == self._swap_intervals:
            return
        self._swap_intervals = wanted
        for window in self._windows:
            window.make_current()
            glfw.swap_interval(1 if wanted[0] and window.index == wanted[1] else 0)
        self._windows[0].make_current()

    def _v_sync_index(self) -> int:
        """The v-sync window's index; window 0 for a name that is not one of the windows."""
        name: str = self.settings.v_sync_window
        return self._window_names.index(name) if name in self._window_names else 0

    # CALLBACKS
    def _on_monitor_change(self, monitor: glfw._GLFWmonitorPointerT, event: int) -> None:
        """GLFW monitor connect/disconnect callback, during poll_events(): handled at the top of the next frame."""
        self._monitors_changed = True

    def _on_key(self, index: int, key: int, action: int, mods: int) -> None:
        """Esc closes; F toggles the fullscreen of the window with focus (applied at the tick)."""
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._close_request.set()
        elif action == glfw.PRESS and key == glfw.KEY_F and 0 <= index < len(self.window_settings):
            settings: WindowSettings = self.window_settings[index]
            settings.fullscreen = not settings.fullscreen

    def _ask_default_placement(self) -> None:
        self._place_default = True

    # PLACEMENT
    def _off_screen(self, settings: WindowSettings) -> bool:
        """A windowed window on a connected monitor, under half of it on the monitors."""
        if settings.fullscreen or not 0 <= settings.monitor < len(self._monitors):
            return False
        monitor: Monitor = self._monitors[settings.monitor]
        return self._visible_share(monitor.x + settings.x, monitor.y + settings.y, settings.width, settings.height) < self._MIN_VISIBLE_SHARE

    def _visible_share(self, x: int, y: int, width: int, height: int) -> float:
        """The share of a window (screen coordinates) that lies on the connected monitors."""
        if width <= 0 or height <= 0:
            return 0.0
        visible: int = 0
        for m in self._monitors:
            w: int = min(x + width, m.x + m.width) - max(x, m.x)
            h: int = min(y + height, m.y + m.height) - max(y, m.y)
            if w > 0 and h > 0:
                visible += w * h
        return visible / (width * height)

    def _write_default_placement(self, settings: WindowSettings) -> None:
        """The default placement into a window's settings: windowed, on its own monitor, its position and size."""
        settings.fullscreen = False
        settings.x = self._DEFAULT_X
        settings.y = self._DEFAULT_Y
        settings.width = self._DEFAULT_WIDTH
        settings.height = self._DEFAULT_HEIGHT

    def _write_connected(self) -> None:
        self.settings.connected = ", ".join(f"M{i} {m.name} {m.width}×{m.height}" for i, m in enumerate(self._monitors))
