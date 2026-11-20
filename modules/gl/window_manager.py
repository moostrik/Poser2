from __future__ import annotations

# Standard library imports
import logging
from enum import IntEnum
from queue import SimpleQueue
from threading import Event, Lock, Thread, current_thread
from time import perf_counter
from typing import Callable, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Third-party imports
from OpenGL.GL import glGetString, GL_VERSION, glBindFramebuffer, GL_FRAMEBUFFER, glViewport
import glfw

# Local application imports
if TYPE_CHECKING:
    from .render_base import RenderBase
from modules.settings import Field, BaseSettings, Widget
from modules.utils import FPS, PerformanceTimer
from .context import Context
from .monitor import Monitor, MonitorId


class FullscreenMode(IntEnum):
    """Mutually exclusive window display modes."""
    WINDOWED            = 0
    WINDOWED_FULLSCREEN = 1  # borderless, monitor resolution


class WindowSettings(BaseSettings):
    """Window / init configuration — set once before WindowManager starts."""
    title: Field[str] =           Field("Poser", access=Field.INIT, visible=False)
    avg_fps: Field[int] =         Field(60, access=Field.READ, pinned=True, label="Render", unit="fps", description="Average render frame rate")
    min_fps: Field[int] =         Field(60, access=Field.READ, pinned=True, label="Render min", unit="fps", description="Minimum render frame rate")
    v_sync: Field[bool] =         Field(True)
    monitor: Field[MonitorId] =   Field(MonitorId.M0, newline=True)
    x: Field[int] =               Field(0)
    y: Field[int] =               Field(80)
    width: Field[int] =           Field(1920)
    height: Field[int] =          Field(1000)
    default_placement: Field[bool] = Field(False, widget=Widget.button, description="Place the window on monitor 0 at 20, 40, 1280 × 720")
    secondary_list: Field[list[MonitorId]] = Field([MonitorId.M0], widget=Widget.order, newline=True)
    secondary_fullscreen: Field[bool] =       Field(True)
    fullscreen_mode: Field[FullscreenMode] = Field(FullscreenMode.WINDOWED)
    fullscreen_mouse: Field[bool] =          Field(False, description="Show the mouse pointer in fullscreen (a window always shows it)")


class MouseButton(IntEnum):
    LEFT    = 0
    RIGHT   = 1
    MIDDLE  = 2
    BACK    = 3
    FORWARD = 4


class WindowManager():
    _MIN_VISIBLE = 100  # px of window that must remain on-screen after a monitor switch
    _MIN_VISIBLE_SHARE = 0.5  # share of the window that must be on the monitors at startup, else the default placement
    # The default placement: monitor 0, relative to it
    _DEFAULT_X = 20
    _DEFAULT_Y = 40
    _DEFAULT_WIDTH = 1280
    _DEFAULT_HEIGHT = 720

    _GLFW_BUTTON_MAP: dict[int, MouseButton] = {
        glfw.MOUSE_BUTTON_LEFT:   MouseButton.LEFT,
        glfw.MOUSE_BUTTON_RIGHT:  MouseButton.RIGHT,
        glfw.MOUSE_BUTTON_MIDDLE: MouseButton.MIDDLE,
        glfw.MOUSE_BUTTON_4:      MouseButton.BACK,
        glfw.MOUSE_BUTTON_5:      MouseButton.FORWARD,
    }

    def __init__(self, renderer: RenderBase, settings: WindowSettings) -> None:
        self.renderer = renderer
        self.settings: WindowSettings = settings

        self._actual_width: int = settings.width
        self._actual_height: int = settings.height
        self._window_width: int = settings.width
        self._window_height: int = settings.height
        self._fullscreen_mode: FullscreenMode = FullscreenMode.WINDOWED
        self._current_vsync: bool | None = None
        logger.info("Initialized with width=%s, height=%s, fullscreen_mode=%s, v_sync=%s, fps=%s", settings.width, settings.height, settings.fullscreen_mode, settings.v_sync, settings.avg_fps)
        self._fps = FPS()
        # CPU time of the renderer's calls; the GPU work they queue is not waited for
        self._update_timer = PerformanceTimer(name="Render update", sample_count=200)
        self._draw_timer = PerformanceTimer(name="Render draw", sample_count=200)
        self._interval_timer = PerformanceTimer(name="Render interval", sample_count=200)
        self._last_update: float = 0.0
        self._mouse_px: float = 0.0
        self._mouse_py: float = 0.0

        self._monitor: Monitor | None = None            # the monitor the main window is on
        self._monitors: list[Monitor] = []               # the connected monitors, in MonitorId order
        self._monitors_changed: bool = False             # a hot-plug, handled at the top of the next frame

        self._loop_thread: Thread | None = None     # the thread that runs the loop, set by run()
        self._close_request = Event()               # stop() asked the loop to end
        self._finished = Event()                    # the loop has ended and cleaned up
        self._finished.set()                        # no loop yet
        self._window_lock = Lock()                  # the main window's handle, for stop() from another thread
        self.frame_time: float = 0.0        # the current frame's stamp (perf_counter), set before its update
        self._callback_lock = Lock()
        # Callbacks run in registration order
        self._exit_callbacks: list[Callable[[], None]] = []
        self._mouse_move_callbacks: list[Callable[[float, float], None]] = []
        self._mouse_button_callbacks: list[Callable[[float, float, MouseButton, bool], None]] = []
        self._mouse_scroll_callbacks: list[Callable[[float], None]] = []
        self._key_callbacks: list[Callable] = []

        # Secondary windows sharing the main context: logical_id → window

        self._main_window: glfw._GLFWwindowPointerT | None = None
        self._secondary_windows: dict[int, glfw._GLFWwindowPointerT] = {}
        self._secondary_fallback: set[int] = set()
        self._monitor_cb = None  # strong ref to monitor callback (prevents GC)
        self._contexts: dict[int, Context] = {}  # id(window) -> its gl.Context, made at its first use

        # Deferred call queue: settings callbacks enqueue GLFW calls here;
        # the render loop drains it on the main thread (GLFW requires main-thread calls).
        self._deferred: SimpleQueue[Callable[[], None]] = SimpleQueue()

        # Reactive bindings: settings → window manager (stored for unbind on teardown)
        s = self.settings
        self._cb_fm         = lambda v: self._deferred.put(lambda: self.set_fullscreen_mode(v))
        self._cb_monitor    = lambda v: self._deferred.put(lambda: self.set_monitor(v))
        self._cb_xy         = lambda _: self._deferred.put(lambda: self.set_position(s.x, s.y))
        self._cb_size       = lambda _: self._deferred.put(lambda: self.set_size(s.width, s.height))
        self._cb_sec_fs     = lambda v: self._deferred.put(lambda: self.set_secondary_fullscreen(v))
        self._cb_mouse      = lambda _: self._deferred.put(self.apply_cursor)
        self._cb_default    = lambda v: self._deferred.put(self.place_default) if v else None
        s.bind(WindowSettings.fullscreen_mode, self._cb_fm)
        s.bind(WindowSettings.monitor,    self._cb_monitor)
        s.bind(WindowSettings.x,         self._cb_xy)
        s.bind(WindowSettings.y,         self._cb_xy)
        s.bind(WindowSettings.width,     self._cb_size)
        s.bind(WindowSettings.height,    self._cb_size)
        s.bind(WindowSettings.secondary_fullscreen, self._cb_sec_fs)
        s.bind(WindowSettings.fullscreen_mouse, self._cb_mouse)
        s.bind(WindowSettings.default_placement, self._cb_default)


    @property
    def fullscreen_mode(self) -> FullscreenMode:
        return self._fullscreen_mode

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
        self.clear_callbacks()
        self._close_request.set()
        with self._window_lock:                         # the loop's cleanup destroys the window under it
            if self._main_window:
                glfw.set_window_should_close(self._main_window, True)     # both allowed from any thread
                glfw.post_empty_event()
        if self._loop_thread is None or current_thread() is self._loop_thread:
            return
        if not self._finished.wait(2.0):
            logger.warning("The window loop did not end within 2 s")

    def _setup(self) -> None:
        """Setup GLFW window and callbacks"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._monitors = Monitor.connected()

        # Place the window before creating it, so it is created at its size
        s = self.settings
        self._monitor = self._logical_monitor(s.monitor)
        if self._monitor is None:
            logger.warning("Monitor %s is not connected (%s monitors), placing the main window on monitor 0", s.monitor, len(self._monitors))
            self._monitor = self._logical_monitor(MonitorId.M0)
            self._write_default_placement()
        else:
            share: float = self._visible_share(self._monitor.x + s.x, self._monitor.y + s.y, s.width, s.height)
            if share < self._MIN_VISIBLE_SHARE:
                logger.warning("Main window is %.0f%% on the monitors, placing it on monitor 0", share * 100)
                self._monitor = self._logical_monitor(MonitorId.M0)
                self._write_default_placement()

        # Create window
        self._main_window = glfw.create_window(s.width, s.height, s.title, None, None)

        if not self._main_window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        posX, posY = self._monitor_pos(self._monitor)
        glfw.set_window_pos(self._main_window, posX + self.settings.x, posY + self.settings.y)

        # Make context current
        glfw.make_context_current(self._main_window)

        version = glGetString(GL_VERSION)
        if isinstance(version, bytes):
            opengl_version: str = version.decode("utf-8")
            logger.info("OpenGL version: %s", opengl_version)
        else:
            raise RuntimeError("OpenGL context is not valid")

        # Set V-Sync
        glfw.swap_interval(1 if self.settings.v_sync else 0)
        self._current_vsync = self.settings.v_sync

        # Set callbacks
        glfw.set_framebuffer_size_callback(self._main_window, self._framebuffer_size_callback)
        glfw.set_window_size_callback(self._main_window, self._window_size_callback)
        glfw.set_window_pos_callback(self._main_window, self._window_pos_callback)
        glfw.set_key_callback(self._main_window, self._notify_key_callback)
        glfw.set_cursor_pos_callback(self._main_window, self._notify_cursor_pos_callback)
        glfw.set_mouse_button_callback(self._main_window, self._notify_mouse_button_callback)
        glfw.set_scroll_callback(self._main_window, self._notify_scroll_callback)

        logger.info("Setting up secondary monitors: %s", self.settings.secondary_list)

        for slot_index, logical_id in enumerate(self.settings.secondary_list):
            win: glfw._GLFWwindowPointerT | None = self._create_secondary_window(f'{logical_id}')
            if not win:
                continue
            self._secondary_windows[logical_id] = win
            monitor: Monitor | None = self._logical_monitor(logical_id)
            if monitor is not None:
                self._setup_secondary_window(win, monitor, self.settings.secondary_fullscreen)
            else:
                logger.info("Monitor slot %s not available at startup, placing in fallback", logical_id)
                self._setup_secondary_window_fallback(win, slot_index)
                self._secondary_fallback.add(logical_id)

        self._monitor_cb = self._on_monitor_change
        glfw.set_monitor_callback(self._monitor_cb)

        glfw.focus_window(self._main_window)

        # Raise secondary windows above the main window so they are in front
        # on any monitor they share with the main window.
        for win in self._secondary_windows.values():
            glfw.focus_window(win)

    def _make_current(self, window: glfw._GLFWwindowPointerT) -> None:
        """`window`'s GL context made current, and its `gl.Context` with it, made at its first use."""
        glfw.make_context_current(window)
        context: Context | None = self._contexts.get(id(window))
        if context is None:
            context = self._contexts[id(window)] = Context()
            context.allocate()
        else:
            context.make_current()

    def _free_context(self, window: glfw._GLFWwindowPointerT) -> None:
        """`window`'s `gl.Context` freed in its own GL context, before the window goes."""
        context: Context | None = self._contexts.pop(id(window), None)
        if context is not None:
            glfw.make_context_current(window)
            context.deallocate()

    def _render_loop(self) -> None:
        """Main rendering loop"""
        assert self._main_window is not None
        self._make_current(self._main_window)  # its context before allocate (shaders may use draw_quad)
        self.renderer.allocate()
        # the starting fullscreen after the renderer is allocated: its resize reaches the renderer at once, in this
        # context, which during the setup would be before the renderer and its context exist
        if self.settings.fullscreen_mode != FullscreenMode.WINDOWED:
            self.set_fullscreen_mode(self.settings.fullscreen_mode)

        try:
            while not self._close_request.is_set() and not glfw.window_should_close(self._main_window):
                # the renderer's objects (FBOs, VAOs) live in the main context: drawing a secondary window
                # leaves its context current, so every frame starts by making the main one current again
                self._make_current(self._main_window)
                if self._monitors_changed:
                    self._monitors_changed = False
                    self._refresh_monitors()
                while not self._deferred.empty():
                    try:
                        self._deferred.get_nowait()()
                    except Exception:
                        logger.exception("Error in deferred call")
                self._update()
                self._draw_main_window()
                for logical_id, win in self._secondary_windows.items():
                    self._draw_secondary_window(logical_id, win)
                try:
                    glfw.poll_events()
                except Exception:
                    # Raised by one of the callbacks (key, mouse, window, monitor): it must not end the loop
                    logger.exception("Error in event callback")
        finally:
            self._make_current(self._main_window)
            self.renderer.deallocate()

    def _update(self) -> None:
        """Update renderer state. Called once per frame; ``frame_time`` is the frame's stamp."""
        self._fps.processed()
        self._push_fps(int(self._fps.get_rate_average()))
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

    def _push_fps(self, fps: int) -> None:
        self.settings.avg_fps = fps
        self.settings.min_fps = int(self._fps.get_rate_minimum())

    def _cleanup(self) -> None:
        """Clean up resources"""
        s = self.settings
        s.unbind_all(self._cb_fm)
        s.unbind_all(self._cb_monitor)
        s.unbind_all(self._cb_xy)
        s.unbind_all(self._cb_size)
        s.unbind_all(self._cb_sec_fs)
        s.unbind_all(self._cb_mouse)
        s.unbind_all(self._cb_default)

        try:
            glfw.set_monitor_callback(None)  # pyright: ignore[reportArgumentType]
            self._monitor_cb = None
            self._destroy_secondary_windows()
            with self._window_lock:
                if self._main_window:
                    self._free_context(self._main_window)
                    glfw.destroy_window(self._main_window)
                    self._main_window = None
                glfw.terminate()
        except Exception as e:
            logger.error("Error cleaning up GLFW: %s", e)

        self._notify_exit_callbacks()

    def _framebuffer_size_callback(self, window: glfw._GLFWwindowPointerT | None, width: int, height: int) -> None:
        """Framebuffer resize (pixels) — used for viewport and renderer."""
        if not window or width <= 0 or height <= 0:
            return
        self._actual_width = width
        self._actual_height = height
        self.renderer.on_main_window_resize(width, height)

    def _window_size_callback(self, window: glfw._GLFWwindowPointerT | None, width: int, height: int) -> None:
        """Window resize (screen coordinates) — write back to settings."""
        if not window or width <= 0 or height <= 0:
            return
        self._window_width = width
        self._window_height = height
        if self._fullscreen_mode == FullscreenMode.WINDOWED:
            self.settings.width = width
            self.settings.height = height

    def _window_pos_callback(self, window: glfw._GLFWwindowPointerT | None, x: int, y: int) -> None:
        """Window move — write back to settings (relative to current monitor)."""
        if not window:
            return
        if self._fullscreen_mode == FullscreenMode.WINDOWED:
            # Track which monitor the window is on as it's dragged across screens
            win_w, win_h = glfw.get_window_size(window)
            cx, cy = x + win_w // 2, y + win_h // 2
            for monitor in self._monitors:
                if monitor.contains(cx, cy):
                    self._monitor = monitor
                    break
            mon_x, mon_y = self._monitor_pos(self._monitor)
            self.settings.x = x - mon_x
            self.settings.y = y - mon_y

    def _draw_main_window(self) -> None:
        assert self._main_window is not None

        self._make_current(self._main_window)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        v_sync = self.settings.v_sync
        if v_sync != self._current_vsync:
            self._current_vsync = v_sync
            glfw.swap_interval(1 if v_sync else 0)

        glViewport(0, 0, self._actual_width, self._actual_height)  # Set viewport here
        start = perf_counter()
        try:
            self.renderer.draw_main(self._actual_width, self._actual_height)
        except Exception:
            logger.exception("Error in draw")
        self._draw_timer.add_time((perf_counter() - start) * 1000.0, report=False)

        glfw.swap_buffers(self._main_window)

    # SECONDARY WINDOWS
    def _create_secondary_window(self, name: str) -> glfw._GLFWwindowPointerT | None:
        if self._main_window is None:
            logger.error("Main window must be created before secondary windows")
            return None

        win: glfw._GLFWwindowPointerT | None = glfw.create_window(100, 100, name, None, self._main_window)
        if not win:
            logger.error("Failed to create secondary window")
            return None

        glfw.make_context_current(win)
        glfw.swap_interval(0)
        self._make_current(self._main_window)
        glfw.set_window_close_callback(win, self._secondary_close_callback)
        glfw.set_key_callback(win, self._notify_key_callback)
        return win

    def _destroy_secondary_windows(self) -> None:
        for win in self._secondary_windows.values():
            self._free_context(win)
            glfw.destroy_window(win)
        self._secondary_windows.clear()

    def _setup_secondary_window(self, window: glfw._GLFWwindowPointerT, monitor: Monitor, fullscreen: bool) -> None:
        """Place a secondary window on `monitor`, fullscreen or windowed"""
        width: int = monitor.width
        height: int = monitor.height
        posX, posY = monitor.x, monitor.y

        if fullscreen:
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_size(window, width, height)
            glfw.set_window_pos(window, posX, posY)
            glfw.set_input_mode(window, glfw.CURSOR, self._cursor_mode(True))
        else:
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_size(window, int(width * 0.5), int(height * 0.5))
            glfw.set_window_pos(window, posX + 100, posY + 100)
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _setup_secondary_window_fallback(self, window: glfw._GLFWwindowPointerT, slot_index: int) -> None:
        """Place a secondary window as a small windowed window on the primary monitor."""
        posX, posY = self._monitor_pos(self._logical_monitor(MonitorId.M0))
        fw, fh = 200, 200
        step: int = fh + 40  # content height + ~30px title bar + 10px gap
        glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
        glfw.set_window_size(window, fw, fh)
        glfw.set_window_pos(window, posX + 40, posY + 40 + slot_index * step)
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _on_monitor_change(self, monitor: glfw._GLFWmonitorPointerT, event: int) -> None:
        """GLFW monitor connect/disconnect callback, during poll_events(): handled at the top of the next frame."""
        self._monitors_changed = True

    def _refresh_monitors(self) -> None:
        """The connected monitors read again after a hot-plug, and the windows placed on them."""
        self._monitors = Monitor.connected()
        if self._monitor is not None and not any(m.is_handle(self._monitor.handle) for m in self._monitors):
            self._monitor = None  # unplugged; the primary monitor stands in until the window moves
        secondary_list = list(self.settings.secondary_list)

        for logical_id, win in self._secondary_windows.items():
            logical_monitor_id = MonitorId(logical_id)
            slot_index: int = secondary_list.index(logical_monitor_id) if logical_monitor_id in secondary_list else 0
            target: Monitor | None = self._logical_monitor(logical_id)

            if target is not None:
                if logical_id in self._secondary_fallback:
                    logger.info("Monitor slot %s now available, moving secondary window to screen", logical_id)
                    self._secondary_fallback.discard(logical_id)
                self._setup_secondary_window(win, target, self.settings.secondary_fullscreen)
                glfw.focus_window(win)
            elif logical_id not in self._secondary_fallback:
                logger.info("Monitor slot %s lost, moving secondary window to fallback", logical_id)
                self._setup_secondary_window_fallback(win, slot_index)
                self._secondary_fallback.add(logical_id)

    def _secondary_close_callback(self, window: glfw._GLFWwindowPointerT) -> None:
        """Close the whole app when any secondary window is closed."""
        assert self._main_window is not None
        glfw.set_window_should_close(self._main_window, True)
        glfw.post_empty_event()

    def _draw_secondary_window(self, logical_id: int, window: glfw._GLFWwindowPointerT) -> None:
        self._make_current(window)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)

        try:
            self.renderer.draw_secondary(logical_id, width, height)
        except Exception:
            logger.exception("Error in draw_secondary")
        glfw.swap_buffers(window)

    # CALLBACKS
    def _notify_key_callback(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                assert self._main_window is not None
                glfw.set_window_should_close(self._main_window, True)
                glfw.post_empty_event()
                return
        if action == glfw.PRESS:
            if key == glfw.KEY_F:
                mode = FullscreenMode.WINDOWED if self._fullscreen_mode == FullscreenMode.WINDOWED_FULLSCREEN else FullscreenMode.WINDOWED_FULLSCREEN
                self.settings.fullscreen_mode = mode

        with self._callback_lock:
            cbs = list(self._key_callbacks)
        for c in cbs:
            c(key, action, mods)

    def _notify_cursor_pos_callback(self, window, xpos, ypos) -> None:
        self._mouse_px = xpos
        self._mouse_py = ypos
        with self._callback_lock:
            cbs = list(self._mouse_move_callbacks)
        for c in cbs:
            c(xpos / self._window_width, ypos / self._window_height)

    def _notify_mouse_button_callback(self, window, button, action, mods) -> None:
        mb = self._GLFW_BUTTON_MAP.get(button)
        if mb is None or action not in (glfw.PRESS, glfw.RELEASE):
            return
        pressed = action == glfw.PRESS
        x = self._mouse_px / self._window_width
        y = self._mouse_py / self._window_height
        with self._callback_lock:
            cbs = list(self._mouse_button_callbacks)
        for c in cbs:
            c(x, y, mb, pressed)

    def _notify_scroll_callback(self, window, xoffset, yoffset) -> None:
        with self._callback_lock:
            cbs = list(self._mouse_scroll_callbacks)
        for c in cbs:
            c(float(yoffset))

    def _notify_exit_callbacks(self) -> None:
        with self._callback_lock:
            cbs = list(self._exit_callbacks)
        for callback in cbs:
            try:
                callback()
            except Exception:
                logger.exception("Error in exit callback")

    def add_mouse_move_callback(self, callback: Callable[[float, float], None]) -> None:
        with self._callback_lock:
            if callback not in self._mouse_move_callbacks:
                self._mouse_move_callbacks.append(callback)

    def add_mouse_button_callback(self, callback: Callable[[float, float, MouseButton, bool], None]) -> None:
        with self._callback_lock:
            if callback not in self._mouse_button_callbacks:
                self._mouse_button_callbacks.append(callback)

    def add_mouse_scroll_callback(self, callback: Callable[[float], None]) -> None:
        """`callback(steps)` for each turn of the wheel over the main window, positive away from the user."""
        with self._callback_lock:
            if callback not in self._mouse_scroll_callbacks:
                self._mouse_scroll_callbacks.append(callback)

    def add_keyboard_callback(self, callback: Callable[[int, int, int], None]) -> None:
        with self._callback_lock:
            if callback not in self._key_callbacks:
                self._key_callbacks.append(callback)

    def add_exit_callback(self, callback) -> None:
        with self._callback_lock:
            if callback not in self._exit_callbacks:
                self._exit_callbacks.append(callback)

    def clear_callbacks(self) -> None:
        with self._callback_lock:
            self._mouse_move_callbacks.clear()
            self._mouse_button_callbacks.clear()
            self._mouse_scroll_callbacks.clear()
            self._key_callbacks.clear()
            # exit_callbacks are intentionally preserved — they must fire on both natural and programmatic stop

    def _cursor_mode(self, fullscreen: bool) -> int:
        """The mouse pointer of a window: hidden in fullscreen unless `fullscreen_mouse` shows it, always shown in a window."""
        return glfw.CURSOR_NORMAL if not fullscreen or self.settings.fullscreen_mouse else glfw.CURSOR_HIDDEN

    def apply_cursor(self) -> None:
        """Set the mouse pointer of every window to what `fullscreen_mouse` says (its callback)."""
        if self._main_window:
            glfw.set_input_mode(self._main_window, glfw.CURSOR, self._cursor_mode(self._fullscreen_mode != FullscreenMode.WINDOWED))
        for logical_id, win in self._secondary_windows.items():
            fullscreen: bool = self.settings.secondary_fullscreen and logical_id not in self._secondary_fallback
            glfw.set_input_mode(win, glfw.CURSOR, self._cursor_mode(fullscreen))

    # SETTERS
    def set_fullscreen_mode(self, mode: FullscreenMode) -> None:
        if not self._main_window: return
        if self._fullscreen_mode == mode: return

        self._fullscreen_mode = mode

        if mode == FullscreenMode.WINDOWED_FULLSCREEN:
            monitor = self._monitor or self._logical_monitor(MonitorId.M0)
            if monitor is None:
                return
            glfw.set_window_attrib(self._main_window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(
                self._main_window, None, monitor.x, monitor.y,  # pyright: ignore[reportArgumentType]
                monitor.width, monitor.height, 0
            )
            glfw.set_input_mode(self._main_window, glfw.CURSOR, self._cursor_mode(True))
        else:  # WINDOWED
            logger.info("Restoring windowed mode")
            mon_x, mon_y = self._monitor_pos(self._monitor)
            glfw.set_window_attrib(self._main_window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_monitor(
                self._main_window, None, mon_x + self.settings.x, mon_y + self.settings.y,  # pyright: ignore[reportArgumentType]
                self.settings.width, self.settings.height, 0
            )
            glfw.set_input_mode(self._main_window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def set_monitor(self, monitor_id: int) -> None:
        """Move the main window to the given monitor (by sorted index)."""
        if not self._main_window:
            return
        monitor = self._logical_monitor(monitor_id)
        if monitor is None:
            return
        self._monitor = monitor
        if self._fullscreen_mode == FullscreenMode.WINDOWED_FULLSCREEN:
            glfw.set_window_monitor(
                self._main_window, None, monitor.x, monitor.y,  # pyright: ignore[reportArgumentType]
                monitor.width, monitor.height, 0
            )
            return
        posX, posY = monitor.x, monitor.y
        max_x = monitor.width - self._MIN_VISIBLE
        max_y = monitor.height - self._MIN_VISIBLE
        x = min(self.settings.x, max_x)
        y = min(self.settings.y, max_y)
        if x != self.settings.x: self.settings.x = x
        if y != self.settings.y: self.settings.y = y
        glfw.set_window_pos(self._main_window, posX + x, posY + y)

    def set_position(self, x: int, y: int) -> None:
        """Move the main window to (x, y) relative to current monitor."""
        if not self._main_window: return
        mon_x, mon_y = self._monitor_pos(self._monitor)
        tx, ty = mon_x + x, mon_y + y
        cx, cy = glfw.get_window_pos(self._main_window)
        if cx == tx and cy == ty: return
        glfw.set_window_pos(self._main_window, tx, ty)

    def set_size(self, width: int, height: int) -> None:
        """Resize the main window."""
        if not self._main_window: return
        if self._window_width == width and self._window_height == height: return
        glfw.set_window_size(self._main_window, width, height)

    def place_default(self) -> None:
        """Place the main window on monitor 0 at the default position and size, windowed."""
        if not self._main_window: return
        monitor = self._logical_monitor(MonitorId.M0)
        self._monitor = monitor
        self._write_default_placement()
        if self.settings.fullscreen_mode != FullscreenMode.WINDOWED:
            self.settings.fullscreen_mode = FullscreenMode.WINDOWED
        if self._fullscreen_mode != FullscreenMode.WINDOWED:
            self.set_fullscreen_mode(FullscreenMode.WINDOWED)  # restores the window at the settings' placement
            return
        mon_x, mon_y = self._monitor_pos(monitor)
        glfw.set_window_size(self._main_window, self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)
        glfw.set_window_pos(self._main_window, mon_x + self._DEFAULT_X, mon_y + self._DEFAULT_Y)

    def _write_default_placement(self) -> None:
        """Write the default placement into the settings: monitor 0, its position and size."""
        s = self.settings
        s.monitor = MonitorId.M0
        s.x = self._DEFAULT_X
        s.y = self._DEFAULT_Y
        s.width = self._DEFAULT_WIDTH
        s.height = self._DEFAULT_HEIGHT

    def _logical_monitor(self, monitor_id: int) -> Monitor | None:
        """The connected monitor at a `MonitorId`, or None when no monitor is there."""
        return self._monitors[monitor_id] if 0 <= monitor_id < len(self._monitors) else None

    @staticmethod
    def _monitor_pos(monitor: Monitor | None) -> tuple[int, int]:
        """A monitor's position on the desktop; (0, 0) without one."""
        return (monitor.x, monitor.y) if monitor is not None else (0, 0)

    def _visible_share(self, x: int, y: int, width: int, height: int) -> float:
        """The share of a window (screen coordinates) that lies on the connected monitors."""
        if width <= 0 or height <= 0: return 0.0
        visible: int = 0
        for m in self._monitors:
            w: int = min(x + width, m.x + m.width) - max(x, m.x)
            h: int = min(y + height, m.y + m.height) - max(y, m.y)
            if w > 0 and h > 0:
                visible += w * h
        return visible / (width * height)

    def set_secondary_fullscreen(self, value: bool) -> None:
        """Set all secondary windows to fullscreen or windowed mode"""

        for logical_id, win in self._secondary_windows.items():
            if logical_id in self._secondary_fallback:
                continue
            monitor: Monitor | None = self._logical_monitor(logical_id)
            if monitor is not None:
                self._setup_secondary_window(win, monitor, value)
