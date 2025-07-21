import OpenGL.GL as gl
import glfw
from threading import Thread, current_thread, Event
from modules.gl.Utils import FpsCounter
from typing import Callable
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
        super().__init__()
        self.window_width: int = width
        self.window_height: int = height
        self.prev_window_width: int = width
        self.prev_window_height: int = height
        # self.running = True
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
        self.mouse_callbacks: list[Callable] = []
        self.key_callbacks: list[Callable] = []

        self.window = None
        self.monitor = None
        self.last_frame_time = 0


        self.render_thread: Thread | None = None
        self._stop_event = Event()
        self.exit_callback: Callable[[], None] | None = None

    def start(self) -> None:
        """Start the rendering thread"""
        self._stop_event.clear()
        self.render_thread = Thread(target=self.run, name="RenderThread")
        self.render_thread.daemon = True
        self.render_thread.start()

    def stop(self) -> None:
        if self.render_thread and self.render_thread.is_alive():
            self._stop_event.set()

            # Multiple ways to signal the render thread to stop
            try:
                if self.window:
                    glfw.set_window_should_close(self.window, True)
                glfw.post_empty_event()  # Wake up glfw.poll_events()
            except:
                pass  # GLFW might not be initialized yet

            if current_thread() != self.render_thread:
                print("Waiting for render thread to finish...")
                self.render_thread.join(timeout=8.0)
                if self.render_thread.is_alive():
                    print("Warning: Render thread did not stop within timeout!")
            print("Render thread stopped.")

    def run(self) -> None:
        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        # Get primary monitor for fullscreen
        self.monitor = glfw.get_primary_monitor() if self.fullscreen else None

        # Create window
        self.window = glfw.create_window(
            self.window_width,
            self.window_height,
            self.windowName,
            None,
            None
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
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_window_close_callback(self.window, self.window_close_callback)

        # Hide cursor in fullscreen
        if self.fullscreen:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        self.initGL()

        version = gl.glGetString(gl.GL_VERSION)
        opengl_version: str = version.decode("utf-8") # type: ignore
        print("OpenGL version:", opengl_version)

        self.last_frame_time = time.time_ns()

        self.allocate()

        # Main loop
        try:
            while not self._stop_event.is_set():
                # Check for window close or stop event
                if glfw.window_should_close(self.window):
                    self.window_close_callback(self.window)
                    break

                current_time = time.time_ns()

                if self.frame_interval:
                    if current_time - self.last_frame_time >= self.frame_interval:
                        self.drawWindow()
                        self.last_frame_time = current_time
                else:
                    self.drawWindow()

                # Non-blocking event polling with timeout
                glfw.poll_events()

                # Small sleep to prevent 100% CPU usage and allow stop event checking
                if not self._stop_event.is_set():
                    time.sleep(0.001)  # 1ms sleep

        except Exception as e:
            print(f"Error in render loop: {e}")

        self.deallocate()


        try:
            if self.window:
                glfw.destroy_window(self.window)
            glfw.terminate()
            print ("GLFW terminated successfully.")
        except Exception as e:
            print(f"Error cleaning up GLFW: {e}")

        # exit the main thread


    def initGL(self) -> None:
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.setView(self.window_width, self.window_height)

    def framebuffer_size_callback(self, window, width, height) -> None:
        self.reshape(width, height)

    def reshape(self, width, height) -> None:
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

        self.draw()

        glfw.swap_buffers(self.window)
        self.fps.tick()

    def window_close_callback(self, window) -> None:
        if self.exit_callback:
            self.exit_callback()

    def draw(self) -> None:
        pass

    def key_callback(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                if self.exit_callback:
                    self.exit_callback()
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

    def cursor_pos_callback(self, window, xpos, ypos) -> None:
        self.mouse_x = xpos / self.window_width
        self.mouse_y = ypos / self.window_height
        for c in self.mouse_callbacks:
            c(self.mouse_x, self.mouse_y, Button.NONE)

    def mouse_button_callback(self, window, button, action, mods) -> None:
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

    def addMouseCallback(self, callback) -> None:
        self.mouse_callbacks.append(callback)

    def clearMouseCallbacks(self) -> None:
        self.mouse_callbacks = []

    def addKeyboardCallback(self, callback) -> None:
        self.key_callbacks.append(callback)

    def clearKeyboardCallbacks(self) -> None:
        self.key_callbacks = []

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
