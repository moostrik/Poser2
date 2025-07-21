import OpenGL.GL as gl
import glfw
from threading import Thread, Lock, current_thread
from modules.gl.Utils import FpsCounter
from typing import Callable, Optional
import time
from enum import Enum


class Button(Enum):
    NONE =          0
    LEFT_UP =       1
    LEFT_DOWN =     2
    MIDDLE_UP =     3
    MIDDLE_DOWN =   4
    RIGHT_UP =      5
    RIGHT_DOWN =    6

class RenderWindow():
    def __init__(self, width, height, name: str, fullscreen: bool = False, v_sync: bool = True, fps: int | None = None, posX = 0, posY = 0) -> None:
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        self.window_width: int = width
        self.window_height: int = height
        self.prev_window_width: int = width
        self.prev_window_height: int = height
        self.fullscreen: bool = fullscreen
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

        self.window = None
        self.monitor = None
        self.last_frame_time = 0

        self.render_thread: Thread | None = None
        self.callback_lock = Lock()
        self.exit_callbacks: set[Callable[[], None]] = set()
        self.mouse_callbacks: set[Callable] = set()
        self.key_callbacks: set[Callable] = set()

    def start(self) -> None:
        """Start the rendering thread"""
        if self.render_thread is None or not self.render_thread.is_alive():
            self.render_thread = Thread(target=self.run, daemon=False)
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
        self.clearCallbacks()
        if self.window:
            glfw.set_window_should_close(self.window, True)
            glfw.post_empty_event()  # Wake up the event loop

        self.render_thread.join(timeout=2.0)  # Wait for thread to finish with timeout
        if self.render_thread.is_alive():
            print(f"Warning: Render thread didn't stop gracefully")
        else:
            print(f"Render thread stopped successfully")

    def run(self) -> None:
        """Main entry point for the render thread"""
        try:
            self._setup_window()
            self._setup_opengl()
            self.allocate()
            self._main_loop()
        except Exception as e:
            print(f"Error in render thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.deallocate()
            self._cleanup()

    def _setup_window(self) -> None:
        """Setup GLFW window and callbacks"""
        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        # Get primary monitor for fullscreen
        self.monitor = glfw.get_primary_monitor() if self.fullscreen else None

        # Create window
        self.window = glfw.create_window(
            self.window_width, self.window_height,
            self.windowName, None, None
        )

        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        # Set window position if not fullscreen
        if not self.fullscreen:
            glfw.set_window_pos(self.window, self.window_x, self.window_y)

        # Make context current
        glfw.make_context_current(self.window)

        # Set V-Sync
        glfw.swap_interval(1 if self.v_sync else 0)

        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self.window_size_callback)
        glfw.set_key_callback(self.window, self.notify_key_callback)
        glfw.set_cursor_pos_callback(self.window, self.notify_cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.notify_invoke_mouse_button_callbacks)

        # Hide cursor in fullscreen
        if self.fullscreen:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    def _setup_opengl(self) -> None:
        """Initialize OpenGL state"""

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.setView(self.window_width, self.window_height)

        version = gl.glGetString(gl.GL_VERSION)
        opengl_version = version.decode("utf-8")  # type: ignore
        print("OpenGL version:", opengl_version)


    def _main_loop(self) -> None:
        """Main rendering loop with improved frame timing"""
        next_frame_time = time.time_ns()
        while not glfw.window_should_close(self.window):

            self.drawWindow()
            glfw.poll_events()

            # Frame timing control
            if not self.v_sync and self.frame_interval:
                next_frame_time += self.frame_interval
                now: int = time.time_ns()
                remaining: int = next_frame_time - now

                if remaining > 0:
                    # Sleep for most of the remaining time
                    sleep_seconds = (remaining - 500_000) / 1_000_000_000  # leave 0.5ms for busy-wait
                    if sleep_seconds > 0.002:
                        time.sleep(sleep_seconds)
                    # Busy-wait for the final bit
                    while time.time_ns() < next_frame_time:
                        pass
                else:
                    if -remaining > self.frame_interval:
                        # If we're behind, reset next_frame_time to now to avoid spiral of death
                        print("Warning: Frame time exceeded by ", -remaining / 1_000_000, "ms")
                        next_frame_time: int = now

    def _cleanup(self) -> None:
        """Clean up resources"""

        try:
            if self.window:
                glfw.destroy_window(self.window)
                self.window = None
            glfw.terminate()
        except Exception as e:
            print(f"Error cleaning up GLFW: {e}")

        self.notify_exit_callbacks()

    def window_size_callback(self, window: Optional[glfw._GLFWwindow], width: int, height: int) -> None:
        if not window or width <= 0 or height <= 0:
            return
        self.window_reshape(width, height, window.title)

    def window_reshape(self, width: int, height: int, name: str) -> None:
        self.window_width = width
        self.window_height = height
        self.setView(width, height)

    def allocate(self) -> None:
        """Allocate resources for the window"""

    def deallocate(self) -> None:
        """Deallocate resources for the window"""

    @staticmethod
    def setView(width, height) -> None:
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, height, 0, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glViewport(0, 0, width, height)

    def setFullscreen(self, value: bool) -> None:
        if not self.window: return
        if self.fullscreen is value: return

        self.fullscreen = value

        if self.fullscreen:
            # Store current window size and position
            self.prev_window_width, self.prev_window_height = glfw.get_window_size(self.window)
            self.window_x, self.window_y = glfw.get_window_pos(self.window)

            # Get monitor and video mode
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)

            # Set fullscreen
            glfw.set_window_monitor(
                self.window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate
            )
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        else:
            # Restore windowed mode
            glfw.set_window_monitor(
                self.window, None, self.window_x, self.window_y,
                self.prev_window_width, self.prev_window_height, 0
            )
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def drawWindow(self) -> None:
        fps = str(self.fps.get_fps())
        min_fps = str(self.fps.get_min_fps())
        glfw.set_window_title(self.window, f'{self.windowName} - FPS: {fps} (Min: {min_fps})')

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
        gl.glLoadIdentity()
        try:
            # Existing code...
            self.draw()
        except Exception as e:
            pass
            print(f"Error in draw: {e}")

        glfw.swap_buffers(self.window)
        self.fps.tick()

    def draw(self) -> None:
        pass

    # CALLBACKS
    def notify_key_callback(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:

                glfw.set_window_should_close(self.window, True)
                glfw.post_empty_event()  # Wake up the event loop


            elif key == glfw.KEY_F:
                self.setFullscreen(not self.fullscreen)
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

    def notify_cursor_pos_callback(self, window, xpos, ypos) -> None:
        self.mouse_x = xpos / self.window_width
        self.mouse_y = ypos / self.window_height
        for c in self.mouse_callbacks:
            c(self.mouse_x, self.mouse_y, Button.NONE)

    def notify_invoke_mouse_button_callbacks(self, window, button, action, mods) -> None:
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

    def notify_exit_callbacks(self) -> None:
        with self.callback_lock:
            for callback in self.exit_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in exit callback: {e}")

    def addMouseCallback(self, callback) -> None:
        with self.callback_lock:
            self.mouse_callbacks.add(callback)

    def addKeyboardCallback(self, callback) -> None:
        with self.callback_lock:
            self.key_callbacks.add(callback)

    def addExitCallback(self, callback) -> None:
        with self.callback_lock:
            self.exit_callbacks.add(callback)

    def clearCallbacks(self) -> None:
        with self.callback_lock:
            self.mouse_callbacks.clear()
            self.key_callbacks.clear()
            self.exit_callbacks.clear()

    @staticmethod
    def draw_string(x: float, y: float, string: str, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0), big: bool = False)-> None:
        # Note: GLFW doesn't have built-in text rendering like GLUT
        # You'll need to implement text rendering using a library like freetype-py or PIL
        # For now, this is a placeholder that sets color but doesn't render text
        gl.glColor4f(*color)
        # TODO: Implement text rendering with freetype-py or similar
        # print(f"Text rendering not implemented: {string} at ({x}, {y})")
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def draw_box_string(x: float, y: float, string: str, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0), box_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6), big: bool = False)-> None:
        # Note: GLFW doesn't have built-in text rendering like GLUT
        # You'll need to implement text rendering using a library like freetype-py or PIL
        height = 18 if big else 12
        expand = 3 if big else 2
        width = len(string) * (12 if not big else 15)  # Rough approximation

        # Draw background box
        gl.glColor4f(*box_color)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x - expand, y - height - expand)
        gl.glVertex2f(x + width + expand, y - height - expand)
        gl.glVertex2f(x + width + expand, y + expand * 2)
        gl.glVertex2f(x - expand, y + expand * 2)
        gl.glEnd()

        gl.glColor4f(*color)
        # TODO: Implement text rendering with freetype-py or similar
        # print(f"Text rendering not implemented: {string} at ({x}, {y})")
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
