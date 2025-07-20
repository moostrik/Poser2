# Standard library imports
import threading
import numpy as np
from threading import Thread, Lock, Event
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any

# Third-party imports, GL etc
import glfw
import OpenGL.GL as gl
from PIL import Image, ImageDraw, ImageFont

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image as GLImage
from modules.gl.Mesh import Mesh
from modules.gl.Shader import Shader

from modules.utils.FPS import FPS

from modules.Settings import Settings


class MultiWindowRender:
    def __init__(self, settings: Settings, on_render_loop_end: Callable[[], None]) -> None:
        self.title: str = settings.render_title
        self.use_vsync: bool = settings.render_v_sync

        self.lock = Lock()
        self.render_thread = None
        self._stop_event = Event()
        self._on_render_loop_end: Callable[[], None] = on_render_loop_end

        self.fps = FPS(120)

    def start(self) -> None:
        """Start the rendering thread"""
        self._stop_event.clear()
        self.render_thread = Thread(target=self._render_loop, name="RenderThread")
        self.render_thread.daemon = True
        self.render_thread.start()

    def stop(self) -> None:
        if self.render_thread and self.render_thread.is_alive():
            self._stop_event.set()

            if threading.current_thread() != self.render_thread:
                self.render_thread.join(timeout=2.0)

    def _render_loop(self):
        """Main rendering thread function"""
        # Initialize GLFW and create windows within this thread
        MultiWindowRender._init_glfw()
        windows: list[Any] = MultiWindowRender._create_windows(self._key_callback)
        # Target frame time (60 FPS default)
        frame_start = time.time()

        while not self._stop_event.is_set():
            last_swap_end = None

            with self.lock:
                for window in windows:
                    glfw.make_context_current(window)
                    # Calculate FPS
                    fps = 1.0 / max(1e-6, time.time() - frame_start)
                    self._render_window(self.fps.get_rate_minimum())

                    # Time since last swap_buffers (between swaps)
                    if last_swap_end is not None:
                        between_swaps = time.perf_counter() - last_swap_end
                    else:
                        between_swaps = 0.0

                    swap_start = time.perf_counter()
                    glfw.swap_buffers(window)
                    swap_end = time.perf_counter()
                    swap_duration = swap_end - swap_start
                    # print(f"swap:{swap_duration*1000:.3f}ms between:{between_swaps*1000:.3f}ms", end=' | ')
                    last_swap_end = swap_end

                glfw.poll_events()

            frame_time = time.time() - frame_start
            print(f"drawloop_total:{frame_time*1000:.3f}ms")
            frame_start = time.time()
            self.fps.processed()
            if not windows:
                break
            time.sleep(0.001)  # Sleep to yield CPU time


        try:
            for window in list(windows):
                glfw.destroy_window(window)
            glfw.terminate()
        except Exception as e:
            print(f"Error cleaning up GLFW: {e}")

        self._on_render_loop_end()


    @ staticmethod
    def _init_glfw() -> None:
        """Initialize GLFW in the render thread"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Configure GLFW
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.VISIBLE, gl.GL_TRUE)

    @ staticmethod
    def _create_windows(callback: Callable) -> list[Any]:
        """
        Create a borderless fullscreen window on each *secondary* monitor, with shared OpenGL context.
        The shared context is created on the first window (first secondary monitor).
        Returns a list of window handles.
        """
        windows = []
        monitors = glfw.get_monitors()
        if not monitors or len(monitors) < 2:
            raise RuntimeError("At least two monitors required to skip the primary monitor.")

        share = None
        # Skip the primary monitor (index 0), use only secondary monitors
        for idx, monitor in enumerate(monitors[1:], start=1):
            mode = glfw.get_video_mode(monitor)
            width = mode.size.width
            height = mode.size.height
            title = f"Window {idx+1}"

            window = glfw.create_window(width, height, title, None, share)
            if not window:
                raise RuntimeError(f"Failed to create borderless window on monitor {idx}")

            # Remove window decorations for borderless effect
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            # Move window to the monitor's position
            xpos, ypos = glfw.get_monitor_pos(monitor)
            glfw.set_window_pos(window, xpos, ypos)
            glfw.make_context_current(window)
            if idx == 1:
                glfw.swap_interval(1)
            else:
                glfw.swap_interval(1)
            glfw.set_key_callback(window, callback)
            glfw.show_window(window)
            windows.append(window)

            # Use the first created window as the context share for others
            if share is None:
                share = window
                # break

        return windows

    @ staticmethod
    def _render_window(fps: float = 0.0):
        """Render specific content for a window - override in subclass"""
        # Clear background
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # --- Render FPS as a texture ---
        img_size = (512, 256)
        img = Image.new("RGBA", img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 128)  # Use a large font
        except:
            font = ImageFont.load_default()
        text = f"{fps:.1f} FPS"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_pos = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
        draw.text(text_pos, text, font=font, fill=(255, 255, 0, 255))

        img_data = img.tobytes("raw", "RGBA", 0, -1)
        tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img_size[0], img_size[1], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # Draw a quad with the texture (legacy OpenGL)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(-0.8, -0.3)
        gl.glTexCoord2f(1, 0); gl.glVertex2f( 0.8, -0.3)
        gl.glTexCoord2f(1, 1); gl.glVertex2f( 0.8,  0.3)
        gl.glTexCoord2f(0, 1); gl.glVertex2f(-0.8,  0.3)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glDeleteTextures([tex_id])

        # --- Draw a moving wide line (as a quad) from left to right ---
        t = time.time()
        xpos = -1.0 + 2.0 * ((t * 0.03) % 1.0)  # Moves from -1.0 to 1.0 over 4 seconds
        width = 0.05  # Adjust for desired line thickness

        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(xpos - width/2, -1.0)
        gl.glVertex2f(xpos + width/2, -1.0)
        gl.glVertex2f(xpos + width/2,  1.0)
        gl.glVertex2f(xpos - width/2,  1.0)
        gl.glEnd()
        gl.glColor3f(1.0, 1.0, 1.0)  # Reset color

    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self._on_render_loop_end()
            # Set window to close
            # glfw.set_window_should_close(window, True)
