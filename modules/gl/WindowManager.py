from __future__ import annotations

# Standard library imports
import logging
from enum import Enum, IntEnum
from threading import Thread, Lock, current_thread
from time import sleep, time_ns
from typing import Callable, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Third-party imports
from OpenGL.GL import (  # type: ignore
    glGetString, GL_VERSION, glBindFramebuffer, GL_FRAMEBUFFER, glViewport,
    glGenBuffers, glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_STATIC_DRAW,
    glGenVertexArrays, glBindVertexArray, glEnableVertexAttribArray,
    glVertexAttribPointer, GL_FLOAT, GL_FALSE, glDeleteVertexArrays, glDeleteBuffers
)
import glfw
import numpy as np
import ctypes

# Local application imports
if TYPE_CHECKING:
    from .RenderBase import RenderBase
from .Utils import FpsCounter
from modules.settings import Field, BaseSettings, Widget


class MonitorId(IntEnum):
    """Logical monitor slots. IntEnum so it's a plain int everywhere."""
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3


class WindowSettings(BaseSettings):
    """Window / init configuration — set once before RenderManager starts."""
    title: Field[str] =           Field("Poser", access=Field.INIT, visible=False)
    avg_fps: Field[int] =         Field(60, access=Field.READ, pinned=True, description="Average Camera FPS")
    min_fps: Field[int] =         Field(60, access=Field.READ, pinned=True, description="Minimum Camera FPS")
    v_sync: Field[bool] =         Field(True)
    fullscreen: Field[bool] =     Field(False)
    monitor: Field[int] =         Field(0, newline=True)
    x: Field[int] =               Field(0)
    y: Field[int] =               Field(80)
    width: Field[int] =           Field(1920)
    height: Field[int] =          Field(1000)
    secondary_list: Field[list[MonitorId]] = Field([MonitorId.M0], widget=Widget.order, newline=True)
    secondary_fullscreen: Field[bool] =  Field(True)


class Button(Enum):
    NONE =          0
    LEFT_UP =       1
    LEFT_DOWN =     2
    MIDDLE_UP =     3
    MIDDLE_DOWN =   4
    RIGHT_UP =      5
    RIGHT_DOWN =    6

class WindowManager():
    def __init__(self, renderer: RenderBase, settings: WindowSettings) -> None:
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        self.renderer = renderer
        self.settings: WindowSettings = settings

        self._actual_width: int = settings.width
        self._actual_height: int = settings.height
        self._window_width: int = settings.width
        self._window_height: int = settings.height
        self.windowed_fullscreen: bool = False
        self.frame_interval: int | None = None
        self._current_vsync: bool | None = None
        logger.info("Initialized with width=%s, height=%s, fullscreen=%s, v_sync=%s, fps=%s", settings.width, settings.height, settings.fullscreen, settings.v_sync, settings.avg_fps)
        self.fps = FpsCounter()
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0

        self.monitor: Optional[glfw._GLFWmonitor] = None
        self._ordered_monitor_ids: list[int] = []

        self.render_thread: Thread | None = None
        self.callback_lock = Lock()
        self.exit_callbacks: set[Callable[[], None]] = set()
        self.mouse_callbacks: set[Callable] = set()
        self.key_callbacks: set[Callable] = set()
        self.fps_callback: Optional[Callable[[int], None]] = None

        # Secondary windows sharing the main context: logical_id → window

        self.main_window: Optional[glfw._GLFWwindow] = None
        self._secondary_windows: dict[int, glfw._GLFWwindow] = {}
        self._secondary_fallback: set[int] = set()
        self._monitor_cb = None  # strong ref to monitor callback (prevents GC)
        self._quad_vaos: dict[int, int] = {}  # id(window) -> VAO
        self._quad_vbos: dict[int, int] = {}  # id(window) -> VBO

        # Reactive bindings: settings → window manager (stored for unbind on teardown)
        s = self.settings
        self._cb_fullscreen = lambda v: self.set_main_fullscreen(v)
        self._cb_monitor    = lambda v: self.set_monitor(v)
        self._cb_xy         = lambda _: self.set_position(s.x, s.y)
        self._cb_size       = lambda _: self.set_size(s.width, s.height)
        self._cb_sec_fs     = lambda v: self.set_secondary_fullscreen(v)
        s.bind(WindowSettings.fullscreen, self._cb_fullscreen)
        s.bind(WindowSettings.monitor,    self._cb_monitor)
        s.bind(WindowSettings.x,         self._cb_xy)
        s.bind(WindowSettings.y,         self._cb_xy)
        s.bind(WindowSettings.width,     self._cb_size)
        s.bind(WindowSettings.height,    self._cb_size)
        s.bind(WindowSettings.secondary_fullscreen, self._cb_sec_fs)

        # FPS feedback: push measured FPS into the readonly settings
        def _push_fps(fps: int) -> None:
            s.avg_fps = fps
            s.min_fps = self.fps.get_min_fps()
        self.fps_callback = _push_fps

    def start(self) -> None:
        """Start the rendering thread"""
        if self.render_thread is None or not self.render_thread.is_alive():
            self.render_thread = Thread(target=self._run, daemon=False)
            self.render_thread.start()

    def stop(self) -> None:
        """Stop the rendering thread"""
        # Early exit if thread doesn't exist or isn't running
        if not self.render_thread or not self.render_thread.is_alive():
            logger.debug("Render thread is not running, nothing to stop")
            return

        # Check if we're calling from the render thread itself
        if current_thread() is self.render_thread:
            logger.debug("Render thread self-terminating")
            return

        # Normal case - external thread stopping the render thread
        self.clear_callbacks()
        if self.main_window:
            glfw.set_window_should_close(self.main_window, True)
            glfw.post_empty_event()  # Wake up the event loop

        self.render_thread.join(timeout=2.0)  # Wait for thread to finish with timeout
        if self.render_thread.is_alive():
            logger.warning("Render thread didn't stop gracefully")

    def _run(self) -> None:
        try:
            self._setup()
            self._render_loop()
        finally:
            self._cleanup()

    def _setup(self) -> None:
        """Setup GLFW window and callbacks"""
        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self._ordered_monitor_ids = self._get_monitors_sorted_by_position()

        # Create window
        self.main_window = glfw.create_window(
            self.settings.width, self.settings.height,
            self.settings.title, None, None
        )

        if not self.main_window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        monitor_id = self.settings.monitor
        monitors = glfw.get_monitors()

        if self._ordered_monitor_ids and 0 <= monitor_id < len(self._ordered_monitor_ids):
            self.monitor = monitors[self._ordered_monitor_ids[monitor_id]]
        else:
            logger.warning("%s ID: %s out of range for available monitors: %s, defaulting to primary monitor", type(self).__name__, monitor_id, len(monitors))
            self.monitor = glfw.get_primary_monitor()

        posX, posY = glfw.get_monitor_pos(self.monitor)
        glfw.set_window_pos(self.main_window, posX + self.settings.x, posY + self.settings.y)


        # Make context current
        glfw.make_context_current(self.main_window)

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
        glfw.set_framebuffer_size_callback(self.main_window, self._framebuffer_size_callback)
        glfw.set_window_size_callback(self.main_window, self._window_size_callback)
        glfw.set_window_pos_callback(self.main_window, self._window_pos_callback)
        glfw.set_key_callback(self.main_window, self._notify_key_callback)
        glfw.set_cursor_pos_callback(self.main_window, self._notify_cursor_pos_callback)
        glfw.set_mouse_button_callback(self.main_window, self._notify_invoke_mouse_button_callbacks)

        logger.info("Setting up secondary monitors: %s", self.settings.secondary_list)

        for slot_index, logical_id in enumerate(self.settings.secondary_list):
            win: glfw._GLFWwindow | None = self._create_secondary_window(f'{logical_id}')
            if not win:
                continue
            self._secondary_windows[logical_id] = win
            if logical_id < len(self._ordered_monitor_ids):
                physical_id: int = self._ordered_monitor_ids[logical_id]
                self._setup_secondary_window(win, physical_id, self.settings.secondary_fullscreen)
            else:
                logger.info("Monitor slot %s not available at startup, placing in fallback", logical_id)
                self._setup_secondary_window_fallback(win, slot_index)
                self._secondary_fallback.add(logical_id)

        self._monitor_cb = self._on_monitor_change
        glfw.set_monitor_callback(self._monitor_cb)

        glfw.focus_window(self.main_window)

        if self.settings.fullscreen:
            self.set_main_fullscreen(True)

    def _create_quad_vao(self) -> tuple[int, int]:
        """Create VAO/VBO for fullscreen quad. Returns (vao, vbo). Call once after context is current."""
        vertices = np.array([
            # x     y     u    v
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ], dtype=np.float32)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return vao, vbo

    def _bind_quad_vao(self, window: glfw._GLFWwindow) -> None:
        """Bind quad VAO for the given window, creating it if needed."""
        key = id(window)
        if key not in self._quad_vaos:
            self._quad_vaos[key], self._quad_vbos[key] = self._create_quad_vao()
        glBindVertexArray(self._quad_vaos[key])

    def _render_loop(self) -> None:
        """Main rendering loop"""
        assert self.main_window is not None
        self._bind_quad_vao(self.main_window)  # Bind VAO before allocate (shaders may use draw_quad)
        self.renderer.allocate()

        next_frame_time: int = time_ns()
        try:
            while not glfw.window_should_close(self.main_window):
                self._update()
                self._draw_main_window()
                for logical_id, win in self._secondary_windows.items():
                    self._draw_secondary_window(logical_id, win)
                glfw.poll_events()

                # Frame timing control (active when v_sync is off and frame_interval is set)
                if not self.settings.v_sync and self.frame_interval:
                    next_frame_time += self.frame_interval
                    now: int = time_ns()
                    remaining: int = next_frame_time - now
                    if remaining > 0:
                        sleep_seconds: float = (remaining - 500_000) / 1_000_000_000
                        if sleep_seconds > 0.002:
                            sleep(sleep_seconds)
                        while time_ns() < next_frame_time:
                            pass
                    elif -remaining > self.frame_interval:
                        next_frame_time = time_ns()  # reset on spiral of death
        finally:
            self.renderer.deallocate()

    def _update(self) -> None:
        """Update renderer state. Called once per frame."""
        self.fps.tick()
        if self.fps_callback:
            self.fps_callback(self.fps.get_fps())
        try:
            self.renderer.update()
        except Exception as e:
            logger.exception("Error in update")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _cleanup(self) -> None:
        """Clean up resources"""
        s = self.settings
        s.unbind_all(self._cb_fullscreen)
        s.unbind_all(self._cb_monitor)
        s.unbind_all(self._cb_xy)
        s.unbind_all(self._cb_size)
        s.unbind_all(self._cb_sec_fs)

        try:
            glfw.set_monitor_callback(None)
            self._monitor_cb = None
            self._destroy_secondary_windows()
            if self.main_window and self._quad_vaos:
                glfw.make_context_current(self.main_window)
                for key, vao in self._quad_vaos.items():
                    glDeleteVertexArrays(1, [vao])
                    if key in self._quad_vbos:
                        glDeleteBuffers(1, [self._quad_vbos[key]])
                self._quad_vaos.clear()
                self._quad_vbos.clear()
            if self.main_window:
                glfw.destroy_window(self.main_window)
                self.main_window = None
            glfw.terminate()
        except Exception as e:
            logger.error("Error cleaning up GLFW: %s", e)

        self._notify_exit_callbacks()

    def _framebuffer_size_callback(self, window: Optional[glfw._GLFWwindow], width: int, height: int) -> None:
        """Framebuffer resize (pixels) — used for viewport and renderer."""
        if not window or width <= 0 or height <= 0:
            return
        self._actual_width = width
        self._actual_height = height
        self.renderer.on_main_window_resize(width, height)

    def _window_size_callback(self, window: Optional[glfw._GLFWwindow], width: int, height: int) -> None:
        """Window resize (screen coordinates) — write back to settings."""
        if not window or width <= 0 or height <= 0:
            return
        self._window_width = width
        self._window_height = height
        if not self.settings.fullscreen and not self.windowed_fullscreen:
            self.settings.width = width
            self.settings.height = height

    def _window_pos_callback(self, window: Optional[glfw._GLFWwindow], x: int, y: int) -> None:
        """Window move — write back to settings (relative to current monitor)."""
        if not window:
            return
        if not self.settings.fullscreen and not self.windowed_fullscreen:
            mon_x, mon_y = glfw.get_monitor_pos(self.monitor) if self.monitor else (0, 0)
            self.settings.x = x - mon_x
            self.settings.y = y - mon_y

    def _draw_main_window(self) -> None:
        assert self.main_window is not None

        glfw.make_context_current(self.main_window)
        v_sync = self.settings.v_sync
        if v_sync != self._current_vsync:
            self._current_vsync = v_sync
            glfw.swap_interval(1 if v_sync else 0)
        self._bind_quad_vao(self.main_window)

        glViewport(0, 0, self._actual_width, self._actual_height)  # Set viewport here
        try:
            self.renderer.draw_main(self._actual_width, self._actual_height)
        except Exception as e:
            logger.exception("Error in draw")

        glfw.swap_buffers(self.main_window)

    # SECONDARY WINDOWS
    def _create_secondary_window(self, name: str) -> Optional[glfw._GLFWwindow]:
        if self.main_window is None:
            logger.error("Main window must be created before secondary windows")
            return None

        win: Optional[glfw._GLFWwindow] = glfw.create_window(100, 100, name, None, self.main_window)
        if not win:
            logger.error("Failed to create secondary window")
            return None

        glfw.make_context_current(win)
        glfw.swap_interval(0)
        glfw.make_context_current(self.main_window)
        glfw.set_window_close_callback(win, self._secondary_close_callback)
        glfw.set_key_callback(win, self._notify_key_callback)
        return win

    def _destroy_secondary_windows(self) -> None:
        for win in self._secondary_windows.values():
            glfw.destroy_window(win)
        self._secondary_windows.clear()

    def _setup_secondary_window(self, window: glfw._GLFWwindow, monitor_id: int, fullscreen: bool) -> None:
        """Setup a secondary window with the given monitor ID and fullscreen mode"""
        monitors: list[glfw._GLFWmonitor] = glfw.get_monitors()
        monitor: Optional[glfw._GLFWmonitor] = monitors[monitor_id] if monitor_id < len(monitors) else None
        if not monitor:
            logger.warning("Monitor %s not found", monitor_id)
            return

        mode: glfw._GLFWvidmode = glfw.get_video_mode(monitor)
        width: int = mode.size.width
        height: int = mode.size.height
        posX, posY = glfw.get_monitor_pos(monitor)

        if fullscreen:
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_size(window, width, height)
            glfw.set_window_pos(window, posX, posY)
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        else:
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_size(window, int(width * 0.5), int(height * 0.5))
            glfw.set_window_pos(window, posX + 100, posY + 100)
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _setup_secondary_window_fallback(self, window: glfw._GLFWwindow, slot_index: int) -> None:
        """Place a secondary window as a small windowed window on the primary monitor."""
        primary: glfw._GLFWmonitor = glfw.get_primary_monitor()
        posX, posY = glfw.get_monitor_pos(primary)
        fw, fh = 200, 200
        step: int = fh + 40  # content height + ~30px title bar + 10px gap
        glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
        glfw.set_window_size(window, fw, fh)
        glfw.set_window_pos(window, posX + 40, posY + 40 + slot_index * step)
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _on_monitor_change(self, monitor: glfw._GLFWmonitor, event: int) -> None:
        """GLFW monitor connect/disconnect callback. Fires on render thread during poll_events()."""
        self._ordered_monitor_ids = self._get_monitors_sorted_by_position()
        secondary_list = list(self.settings.secondary_list)

        for logical_id, win in self._secondary_windows.items():
            logical_monitor_id = MonitorId(logical_id)
            slot_index: int = secondary_list.index(logical_monitor_id) if logical_monitor_id in secondary_list else 0
            target_available: bool = logical_id < len(self._ordered_monitor_ids)

            if target_available and logical_id in self._secondary_fallback:
                physical_id: int = self._ordered_monitor_ids[logical_id]
                logger.info("Monitor slot %s now available, moving secondary window to screen", logical_id)
                self._setup_secondary_window(win, physical_id, self.settings.secondary_fullscreen)
                self._secondary_fallback.discard(logical_id)
            elif not target_available and logical_id not in self._secondary_fallback:
                logger.info("Monitor slot %s lost, moving secondary window to fallback", logical_id)
                self._setup_secondary_window_fallback(win, slot_index)
                self._secondary_fallback.add(logical_id)

    def _secondary_close_callback(self, window: glfw._GLFWwindow) -> None:
        """Close the whole app when any secondary window is closed."""
        glfw.set_window_should_close(self.main_window, True)
        glfw.post_empty_event()

    def _draw_secondary_window(self, logical_id: int, window: glfw._GLFWwindow) -> None:
        glfw.make_context_current(window)  # <-- Make this window's context current
        self._bind_quad_vao(window)
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)

        try:
            self.renderer.draw_secondary(logical_id, width, height)
        except Exception as e:
            logger.exception("Error in draw_secondary")
        glfw.swap_buffers(window)

    @staticmethod
    def _get_monitors_sorted_by_position() -> list[int]:
        monitors: list[glfw._GLFWmonitor] = glfw.get_monitors()
        if not monitors:
            return []

        primary_monitor = glfw.get_primary_monitor()
        primary_idx: int = next((i for i, m in enumerate(monitors) if m == primary_monitor), 0)

        # Pair each non-primary monitor index with its (x, y) position
        others_with_pos: list[tuple[int, int, int]] = [
            (i, *glfw.get_monitor_pos(m)) for i, m in enumerate(monitors) if i != primary_idx
        ]
        # Sort by x, then by y (negative y before positive y)
        others_sorted: list[tuple[int, int, int]] = sorted(
            others_with_pos,
            key=lambda item: (item[1], item[2])
        )

        return [primary_idx] + [idx for idx, _, _ in others_sorted]

    # CALLBACKS
    def _notify_key_callback(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:

                glfw.set_window_should_close(self.main_window, True)
                glfw.post_empty_event()  # Wake up the event loop

            elif key == glfw.KEY_F:
                self.settings.fullscreen = not self.settings.fullscreen
                return
            elif key == glfw.KEY_W:
                self.set_main_windowed_fullscreen(not self.windowed_fullscreen)
                return

            # Convert GLFW key to byte for compatibility with existing callbacks
            key_byte = None
            if 32 <= key <= 126:  # Printable ASCII characters
                key_byte = bytes([key])
            elif key == glfw.KEY_ESCAPE:
                key_byte = b'\x1b'

            if key_byte:
                mx = self.mouse_x * self._actual_width if window is self.main_window else 0.0
                my = self.mouse_y * self._actual_height if window is self.main_window else 0.0
                with self.callback_lock:
                    cbs = list(self.key_callbacks)
                for c in cbs:
                    c(key_byte, mx, my)

    def _notify_cursor_pos_callback(self, window, xpos, ypos) -> None:
        self.mouse_x = xpos / self._window_width
        self.mouse_y = ypos / self._window_height
        with self.callback_lock:
            cbs = list(self.mouse_callbacks)
        for c in cbs:
            c(self.mouse_x, self.mouse_y, Button.NONE)

    def _notify_invoke_mouse_button_callbacks(self, window, button, action, mods) -> None:
        button_enum: Button = Button.NONE

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                button_enum = Button.LEFT_DOWN
            elif action == glfw.RELEASE:
                button_enum = Button.LEFT_UP
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            if action == glfw.PRESS:
                button_enum = Button.MIDDLE_DOWN
            elif action == glfw.RELEASE:
                button_enum = Button.MIDDLE_UP
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                button_enum = Button.RIGHT_DOWN
            elif action == glfw.RELEASE:
                button_enum = Button.RIGHT_UP

        with self.callback_lock:
            cbs = list(self.mouse_callbacks)
        for c in cbs:
            c(self.mouse_x, self.mouse_y, button_enum)

    def _notify_exit_callbacks(self) -> None:
        with self.callback_lock:
            cbs = list(self.exit_callbacks)
        for callback in cbs:
            try:
                callback()
            except Exception as e:
                logger.exception("Error in exit callback")

    def add_mouse_callback(self, callback) -> None:
        with self.callback_lock:
            self.mouse_callbacks.add(callback)

    def add_keyboard_callback(self, callback) -> None:
        with self.callback_lock:
            self.key_callbacks.add(callback)

    def add_exit_callback(self, callback) -> None:
        with self.callback_lock:
            self.exit_callbacks.add(callback)

    def clear_callbacks(self) -> None:
        with self.callback_lock:
            self.mouse_callbacks.clear()
            self.key_callbacks.clear()
            # exit_callbacks are intentionally preserved — they must fire on both natural and programmatic stop

    # SETTERS
    def set_main_fullscreen(self, value: bool) -> None:
        if not self.main_window: return

        if value:
            self.set_main_windowed_fullscreen(False)

            ordered_ids = self._ordered_monitor_ids
            monitors = glfw.get_monitors()
            monitor_index = ordered_ids[self.settings.monitor] if self.settings.monitor < len(ordered_ids) else ordered_ids[0] if ordered_ids else 0
            monitor = monitors[monitor_index]
            mode = glfw.get_video_mode(monitor)

            glfw.set_window_monitor(
                self.main_window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate
            )
            glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        else:
            if self.windowed_fullscreen:
                return  # set_main_windowed_fullscreen is driving the transition
            # Restore windowed mode from settings (preserved during fullscreen)
            mon_x, mon_y = glfw.get_monitor_pos(self.monitor) if self.monitor else (0, 0)
            glfw.set_window_monitor(
                self.main_window, None, mon_x + self.settings.x, mon_y + self.settings.y,
                self.settings.width, self.settings.height, 0
            )
            glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def set_main_windowed_fullscreen(self, value: bool) -> None:
        if not self.main_window: return
        if self.windowed_fullscreen is value: return

        self.windowed_fullscreen = value

        if self.windowed_fullscreen:
            self.settings.fullscreen = False  # exit regular fullscreen if active

            # Get monitor and video mode
            monitor = self.monitor or glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            posX, posY = glfw.get_monitor_pos(monitor)

            # Set windowed fullscreen (borderless, monitor resolution)
            glfw.set_window_attrib(self.main_window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(
                self.main_window, None, posX, posY,
                mode.size.width, mode.size.height, 0
            )
        else:
            # Restore windowed mode from settings (preserved during fullscreen)
            logger.info("Restoring windowed mode")
            mon_x, mon_y = glfw.get_monitor_pos(self.monitor) if self.monitor else (0, 0)
            glfw.set_window_attrib(self.main_window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_monitor(
                self.main_window, None, mon_x + self.settings.x, mon_y + self.settings.y,
                self.settings.width, self.settings.height, 0
            )

    def set_monitor(self, monitor_id: int) -> None:
        """Move the main window to the given monitor (by sorted index)."""
        if not self.main_window:
            return
        ordered_ids = self._ordered_monitor_ids
        if monitor_id < 0 or monitor_id >= len(ordered_ids):
            return
        real_id = ordered_ids[monitor_id]
        monitors = glfw.get_monitors()
        if real_id >= len(monitors):
            return
        self.monitor = monitors[real_id]
        posX, posY = glfw.get_monitor_pos(self.monitor)
        glfw.set_window_pos(self.main_window, posX + self.settings.x, posY + self.settings.y)

    def set_position(self, x: int, y: int) -> None:
        """Move the main window to (x, y) relative to current monitor."""
        if not self.main_window:
            return
        mon_x, mon_y = glfw.get_monitor_pos(self.monitor) if self.monitor else (0, 0)
        glfw.set_window_pos(self.main_window, mon_x + x, mon_y + y)

    def set_size(self, width: int, height: int) -> None:
        """Resize the main window."""
        if not self.main_window:
            return
        glfw.set_window_size(self.main_window, width, height)

    def set_secondary_fullscreen(self, value: bool) -> None:
        """Set all secondary windows to fullscreen or windowed mode"""

        ordered_ids = self._ordered_monitor_ids
        for logical_id, win in self._secondary_windows.items():
            if logical_id in self._secondary_fallback:
                continue
            if logical_id >= len(ordered_ids):
                continue
            physical_id: int = ordered_ids[logical_id]
            self._setup_secondary_window(win, physical_id, value)
