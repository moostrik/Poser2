# Standard library imports
from enum import Enum
from threading import Thread, Lock, current_thread
from time import sleep, time_ns
import traceback
from typing import Callable, Optional

# Third-party imports
from OpenGL.GL import glGetString, GL_VERSION
import glfw

# Local application imports
from modules.gl.RenderBase import RenderBase
from modules.gl.Utils import FpsCounter, invalidate_quad_cache


class Button(Enum):
    NONE =          0
    LEFT_UP =       1
    LEFT_DOWN =     2
    MIDDLE_UP =     3
    MIDDLE_DOWN =   4
    RIGHT_UP =      5
    RIGHT_DOWN =    6

class WindowManager():
    def __init__(self, renderer: RenderBase, width, height, name: str, fullscreen: bool = False, v_sync: bool = True, fps: int | None = None, posX = 0, posY = 0, monitor_id = 0, secondary_monitor_ids: list[int] = []) -> None:
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        self.renderer = renderer

        self.window_width: int = width
        self.window_height: int = height
        self.prev_window_width: int = width
        self.prev_window_height: int = height
        self.fullscreen: bool = fullscreen
        self.windowed_fullscreen: bool = False
        self.v_sync: bool = v_sync
        self.frame_interval: None | int = None
        if fps and fps > 0:
            self.frame_interval = int((1.0 / fps) * 1_000_000_000)
            self.v_sync = False  # Disable v-sync if we are controlling FPS manually
        self.windowName: str = name
        self.fps = FpsCounter()
        self.window_x: int = posX
        self.window_y: int = posY
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0

        self.monitor: Optional[glfw._GLFWmonitor] = None
        self._ordered_monitor_ids: list[int] = []
        self.monitor_id: int = monitor_id

        self.render_thread: Thread | None = None
        self.callback_lock = Lock()
        self.exit_callbacks: set[Callable[[], None]] = set()
        self.mouse_callbacks: set[Callable] = set()
        self.key_callbacks: set[Callable] = set()

        # List of secondary windows sharing the main context

        self.main_window: Optional[glfw._GLFWwindow] = None
        self.secondary_monitor_ids: list[int] = secondary_monitor_ids
        self.secondary_windows: list[glfw._GLFWwindow] = []
        self.secondary_fullscreen = True

    def start(self) -> None:
        """Start the rendering thread"""
        if self.render_thread is None or not self.render_thread.is_alive():
            self.render_thread = Thread(target=self._run, daemon=False)
            self.render_thread.start()

    def stop(self) -> None:
        """Stop the rendering thread"""
        # Early exit if thread doesn't exist or isn't running
        if not self.render_thread or not self.render_thread.is_alive():
            print(f"Render thread is not running, nothing to stop")
            return

        # Check if we're calling from the render thread itself
        if current_thread() is self.render_thread:
            print(f"Render thread self-terminating")
            return

        # Normal case - external thread stopping the render thread
        self.clear_callbacks()
        if self.main_window:
            glfw.set_window_should_close(self.main_window, True)
            glfw.post_empty_event()  # Wake up the event loop

        self.render_thread.join(timeout=2.0)  # Wait for thread to finish with timeout
        if self.render_thread.is_alive():
            print(f"Warning: Render thread didn't stop gracefully")

    def _run(self) -> None:
        self._setup()
        self._render_loop()
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

        # Get primary monitor for fullscreen
        self.monitor: Optional[glfw._GLFWmonitor] = glfw.get_primary_monitor() if self.fullscreen else None

        # Create window
        self.main_window = glfw.create_window(
            self.window_width, self.window_height,
            self.windowName, None, None
        )

        if not self.main_window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        if self.monitor_id > len(self._ordered_monitor_ids):
            self.monitor_id = len(self._ordered_monitor_ids) - 1

        monitors = glfw.get_monitors()

        if self.monitor_id < 0 or self.monitor_id >= len(monitors):
            print(f"{self.__class__.__name__} ID: {self.monitor_id} out of range for available monitors: {len(monitors)}, defaulting to primary monitor")
            self.monitor_id = 0

        self.monitor = monitors[self._ordered_monitor_ids[self.monitor_id]]

        mode: glfw._GLFWvidmode = glfw.get_video_mode(self.monitor)
        width: int = mode.size.width
        height: int = mode.size.height
        posX, posY = glfw.get_monitor_pos(self.monitor)
        self.window_x += posX
        self.window_y += posY

        glfw.set_window_pos(self.main_window, self.window_x, self.window_y)


        # Make context current
        glfw.make_context_current(self.main_window)

        version = glGetString(GL_VERSION)
        if isinstance(version, bytes):
            opengl_version: str = version.decode("utf-8")
            print("OpenGL version:", opengl_version)
        else:
            raise RuntimeError("OpenGL context is not valid")

        # Set V-Sync
        glfw.swap_interval(1 if self.v_sync else 0)

        # Set callbacks
        glfw.set_framebuffer_size_callback(self.main_window, self._window_size_callback)
        glfw.set_key_callback(self.main_window, self._notify_key_callback)
        glfw.set_cursor_pos_callback(self.main_window, self._notify_cursor_pos_callback)
        glfw.set_mouse_button_callback(self.main_window, self._notify_invoke_mouse_button_callbacks)

        print(f"Setting up secondary monitors: {self.secondary_monitor_ids}")

        for id in self.secondary_monitor_ids:
            if id < 0 or id >= len(self._ordered_monitor_ids):
                print(f"{self.__class__.__name__} ID: {id} out of range for available monitors: {self._ordered_monitor_ids}")
                continue

            monitor_id: int = self._ordered_monitor_ids[id]
            name: str = f'{id}'
            win: glfw._GLFWwindow | None = self._create_secondary_window(name, monitor_id)
            if win:
                self.secondary_windows.append(win)

        glfw.focus_window(self.main_window)

        if self.fullscreen:
            self.set_main_windowed_fullscreen(True)

    def _render_loop(self) -> None:
        """Main rendering loop with improved frame timing"""
        self.renderer.allocate()

        next_frame_time = time_ns()
        while not glfw.window_should_close(self.main_window):

            self._draw_main_window()
            for win in self.secondary_windows:
                self._draw_secondary_window(win)
            glfw.poll_events()

            # Frame timing control
            if not self.v_sync and self.frame_interval:
                next_frame_time += self.frame_interval
                now: int = time_ns()
                remaining: int = next_frame_time - now

                if remaining > 0:
                    # Sleep for most of the remaining time
                    sleep_seconds: float = (remaining - 500_000) / 1_000_000_000  # leave 0.5ms for busy-wait
                    if sleep_seconds > 0.002:
                        sleep(sleep_seconds)
                    # Busy-wait for the final bit
                    while time_ns() < next_frame_time:
                        pass
                else:
                    if -remaining > self.frame_interval:
                        # If we're behind, reset next_frame_time to now to avoid spiral of death
                        # print("Warning: Frame time exceeded by ", -remaining / 1_000_000, "ms")
                        next_frame_time: int = now


        self.renderer.deallocate()

    def _cleanup(self) -> None:
        """Clean up resources"""

        try:
            self._destroy_secondary_windows()
            if self.main_window:
                glfw.destroy_window(self.main_window)
                self.main_window = None
            glfw.terminate()
        except Exception as e:
            print(f"Error cleaning up GLFW: {e}")

        self._notify_exit_callbacks()

    def _window_size_callback(self, window: Optional[glfw._GLFWwindow], width: int, height: int) -> None:
        if not window or width <= 0 or height <= 0:
            return
        self.window_width = width
        self.window_height = height
        self.renderer.on_main_window_resize(width, height)

    def _draw_main_window(self) -> None:
        fps = str(self.fps.get_fps())
        min_fps = str(self.fps.get_min_fps())
        glfw.set_window_title(self.main_window, f'{self.windowName} - FPS: {fps} (Min: {min_fps})')


        glfw.make_context_current(self.main_window)
        invalidate_quad_cache()
        glfw.swap_interval(1 if self.v_sync else 0)

        try:
            self.renderer.draw_main(self.window_width, self.window_height)
        except Exception as e:
            print(f"Error in draw: {e}")
            traceback.print_exc()

        glfw.swap_buffers(self.main_window)
        self.fps.tick()

    # SECONDARY WINDOWS
    def _create_secondary_window(self, name:str, monitor_id: int) -> Optional[glfw._GLFWwindow]:
        if self.main_window is None:
            print("Main window must be created before secondary windows.")
            return None

        monitors: list[glfw._GLFWmonitor] = glfw.get_monitors()
        if monitor_id < 0 or monitor_id >= len(monitors):
            print(f"Monitor ID {monitor_id} is out of range for available monitors: {len(monitors)}")
            return None

        monitor: glfw._GLFWmonitor = monitors[monitor_id]

        win: Optional[glfw._GLFWwindow] = glfw.create_window(100, 100, name, None, self.main_window)
        if not win:
            print("Failed to create secondary window")
            return None

        glfw.set_mouse_button_callback(win, self._invoke_secondary_callbacks)
        self._setup_secondary_window(win, monitor_id, True)

        return win

    def _destroy_secondary_windows(self) -> None:
        for win in self.secondary_windows:
            glfw.destroy_window(win)
        self.secondary_windows.clear()

    def _invoke_secondary_callbacks(self, window, button, action, mods) -> None:
        # Notify the appropriate callbacks for the secondary window
        # only on mouse_down
        if action == glfw.PRESS:
            self.secondary_fullscreen: bool = not self.secondary_fullscreen
            self.set_secondary_fullscreen(self.secondary_fullscreen)

    def _setup_secondary_window(self, window: glfw._GLFWwindow, monitor_id: int, fullscreen: bool) -> None:
        """Setup a secondary window with the given monitor ID and fullscreen mode"""
        monitor: Optional[glfw._GLFWmonitor] = glfw.get_monitors()[monitor_id] if monitor_id < len(glfw.get_monitors()) else None
        if not monitor:
            print(f"Monitor {monitor_id} not found")
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

    def _draw_secondary_window(self, window: glfw._GLFWwindow) -> None:
        title: Optional[str] | None = glfw.get_window_title(window)
        monitor_id: int = 0
        if title:
            monitor_id = int(title)

        width, height = glfw.get_window_size(window)

        glfw.make_context_current(window)  # <-- Make this window's context current
        invalidate_quad_cache()
        glfw.swap_interval(0)
        try:
            self.renderer.draw_secondary(monitor_id, width, height)
        except Exception as e:
            print(f"Error in draw_secondary: {e}")
            traceback.print_exc()  # This prints the stack trace
        glfw.swap_buffers(window)

    @staticmethod
    def _get_monitors_sorted_by_position() -> list[int]:
        monitors: list[glfw._GLFWmonitor] = glfw.get_monitors()
        if not monitors:
            return []

        primary = 0
        others: list[glfw._GLFWmonitor] = monitors[1:]

        # Pair each monitor index with its (x, y) position
        others_with_pos: list[tuple[int, int, int]] = [
            (i + 1, *glfw.get_monitor_pos(monitor)) for i, monitor in enumerate(others)
        ]
        # Sort by x, then by y (negative y before positive y)
        others_sorted: list[tuple[int, int, int]] = sorted(
            others_with_pos,
            key=lambda item: (item[1], item[2])
        )

        return [primary] + [idx for idx, _, _ in others_sorted]

    # CALLBACKS
    def _notify_key_callback(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:

                glfw.set_window_should_close(self.main_window, True)
                glfw.post_empty_event()  # Wake up the event loop

            elif key == glfw.KEY_F:
                self.set_main_fullscreen(not self.fullscreen)
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
                for c in self.key_callbacks:
                    c(key_byte, self.mouse_x * self.window_width, self.mouse_y * self.window_height)

    def _notify_cursor_pos_callback(self, window, xpos, ypos) -> None:
        self.mouse_x = xpos / self.window_width
        self.mouse_y = ypos / self.window_height
        for c in self.mouse_callbacks:
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

        for c in self.mouse_callbacks:
            c(self.mouse_x, self.mouse_y, button_enum)

    def _notify_exit_callbacks(self) -> None:
        with self.callback_lock:
            for callback in self.exit_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in exit callback: {e}")

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
            self.exit_callbacks.clear()

    # SETTERS
    def set_main_fullscreen(self, value: bool) -> None:
        if not self.main_window: return
        if self.fullscreen is value: return

        self.fullscreen = value

        if self.fullscreen:
            self.set_main_windowed_fullscreen(False)
            # Store current window size and position
            self.prev_window_width, self.prev_window_height = glfw.get_window_size(self.main_window)
            self.window_x, self.window_y = glfw.get_window_pos(self.main_window)

            monitors = glfw.get_monitors()
            monitor_index = self._ordered_monitor_ids[self.monitor_id] if self.monitor_id < len(self._ordered_monitor_ids) else 0
            monitor = monitors[monitor_index]
            mode = glfw.get_video_mode(monitor)

            # Set fullscreen
            glfw.set_window_monitor(
                self.main_window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate
            )
            glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        else:
            # Restore windowed mode
            glfw.set_window_monitor(
                self.main_window, None, self.window_x, self.window_y,
                self.prev_window_width, self.prev_window_height, 0
            )
            glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def set_main_windowed_fullscreen(self, value: bool) -> None:
        if not self.main_window: return
        if self.windowed_fullscreen is value: return

        self.windowed_fullscreen = value

        if self.windowed_fullscreen:
            self.set_main_fullscreen(False)

            # Store current window size and position
            self.prev_window_width, self.prev_window_height = glfw.get_window_size(self.main_window)
            self.window_x, self.window_y = glfw.get_window_pos(self.main_window)

            # Get monitor and video mode
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)

            # Set fullscreen
            glfw.set_window_attrib(self.main_window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_monitor(
                self.main_window, None, self.window_x, self.window_y,
                mode.size.width, mode.size.height, 0
            )
            # glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        else:
            # Restore windowed mode
            print("Restoring windowed mode")
            glfw.set_window_attrib(self.main_window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_monitor(
                self.main_window, None, self.window_x, self.window_y,
                self.prev_window_width, self.prev_window_height, 0
            )
            # glfw.set_input_mode(self.main_window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def set_secondary_fullscreen(self, value: bool) -> None:
        """Set all secondary windows to fullscreen or windowed mode"""

        for win in self.secondary_windows:
            title: Optional[str] | None = glfw.get_window_title(win)
            if title is None:
                continue
            id = int(title)
            monitor_id: int = self._ordered_monitor_ids[id] if id < len(self._ordered_monitor_ids) else 0
            monitor: Optional[glfw._GLFWmonitor] = glfw.get_monitors()[monitor_id] if monitor_id < len(glfw.get_monitors()) else None
            if not monitor:
                continue
            if value:
                self._setup_secondary_window(win, monitor_id, True)
            else:
                self._setup_secondary_window(win, monitor_id, False)
