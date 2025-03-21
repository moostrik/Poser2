from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT
from threading import Thread
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

class RenderWindow(Thread):
    def __init__(self, width, height, name: str, fullscreen: bool = False, v_sync: bool = True, fps: int | None = None, posX = 0, posY = 0) -> None:
        super().__init__()
        self.window_width: int = width
        self.window_height: int = height
        self.prev_window_width: int = width
        self.prev_window_height: int = height
        self.running = True
        self.fullscreen: bool = fullscreen
        self.v_sync: bool = v_sync
        # self.frame_interval = int(1000 / fps)  # Interval in milliseconds
        self.frame_interval: None | int = None
        if fps:
            self.frame_interval = int((1.0 / fps) * 1_000_000_000)
        self.last_time: int = time.time_ns()
        self.windowName: str = name
        self.fps = FpsCounter()
        self.window_x: int = posX
        self.window_y: int = posY
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0
        self.mouse_callbacks: list[Callable] = []
        self.key_callbacks: list[Callable] = []

        self._is_allocated: bool = False
        self.exit_callback: Callable | None = None

    @property
    def is_allocated(self)-> bool:
        return self._is_allocated


    def run(self) -> None:
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH) # type: ignore
        glut.glutInitWindowSize(self.window_width, self.window_height)
        glut.glutInitWindowPosition(self.window_x, self.window_y)
        glut.glutCreateWindow(self.windowName)
        if self.fullscreen :
            glut.glutFullScreen()
            glut.glutSetCursor(glut.GLUT_CURSOR_NONE)
        if self.v_sync:
            wglSwapIntervalEXT(1)
        self.initGL()
        glut.glutDisplayFunc(self.drawWindow)
        glut.glutIdleFunc(self.drawWindow)
        glut.glutKeyboardFunc(self.keyboardCallback)
        glut.glutSpecialFunc(self.specialKeyCallback)
        glut.glutPassiveMotionFunc(self.mouseMotionCallback)
        glut.glutMotionFunc(self.mouseMotionCallback) # on mouse button down
        glut.glutMouseFunc(self.mouseButtonCallback)  # Register mouse button callback
        glut.glutReshapeFunc(self.reshape)
        glut.glutCloseFunc(self.closeWindow)

        version = glGetString(GL_VERSION)
        opengl_version: str = version.decode("utf-8") # type: ignore
        print("OpenGL version:", opengl_version)

        glut.glutMainLoop()

    def stop(self) -> None:
        if self._is_allocated:
            glut.glutLeaveMainLoop()

    def initGL(self) -> None:
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.setView(self.window_width, self.window_height)

    def reshape(self, width, height) -> None:
        self.window_width = width
        self.window_height = height
        self.setView(self.window_width, self.window_height)

    def setView(self, width, height) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)

    def setFullscreen(self, value: bool) -> None:
        if not self.is_allocated: return
        if self.fullscreen is value: return
        self.fullscreen = value

        if self.fullscreen:
            self.prev_window_width = self.window_width
            self.prev_window_height = self.window_height
            glut.glutFullScreen()
            # glut.glutSetCursor(glut.GLUT_CURSOR_NONE)
        else:
            glut.glutReshapeWindow(self.prev_window_width, self.prev_window_height)
            glut.glutPositionWindow(100, 100)
            # glut.glutSetCursor(glut.GLUT_CURSOR_LEFT_ARROW)

    def drawWindow(self) -> None:
        current_time: int = time.time_ns()
        elapsed_time: int = current_time - self.last_time

        if self.frame_interval and elapsed_time < self.frame_interval:
            time.sleep((self.frame_interval - elapsed_time) / 1_000_000_000)
        self.last_time = time.time_ns()

        self.fps.tick()
        fps = str(self.fps.get_fps())
        glut.glutSetWindowTitle(' FPS: ' + fps + '          ' + self.windowName)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore
        glLoadIdentity()

        self.draw()

        glut.glutSwapBuffers()

        self._is_allocated = True

    def closeWindow(self,) -> None:
        self._is_allocated = False
        if self.exit_callback: self.exit_callback()

    def draw(self) -> None:
        pass

    def keyboardCallback(self, key, x, y) -> None:
        if key == b'\x1b': # escape
            if self._is_allocated: # redundant?
                glut.glutLeaveMainLoop()
        for c in self.key_callbacks:
            c(key, x, y)

    def specialKeyCallback(self, key, x, y) -> None:
        pass

    def mouseMotionCallback(self, x: int, y: int) -> None:
        self.mouse_x = x / self.window_width
        self.mouse_y = y / self.window_height
        for c in self.mouse_callbacks:
            c(self.mouse_x, self.mouse_y, Button.NONE)

    def mouseButtonCallback(self, button, state, x, y) -> None:
        button_enum: Button = Button.NONE

        if button == glut.GLUT_LEFT_BUTTON:
            if state == glut.GLUT_DOWN:
                button_enum = Button.LEFT_DOWN
            elif state == glut.GLUT_UP:
                button_enum = Button.LEFT_UP
        elif button == glut.GLUT_MIDDLE_BUTTON:
            if state == glut.GLUT_DOWN:
                button_enum = Button.MIDDLE_DOWN
            elif state == glut.GLUT_UP:
                button_enum = Button.MIDDLE_UP
        elif button == glut.GLUT_RIGHT_BUTTON:
            if state == glut.GLUT_DOWN:
                button_enum = Button.RIGHT_DOWN
            elif state == glut.GLUT_UP:
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


    def draw_string(self, x: int, y: int, string: str, flipV=False):
        glRasterPos2f(x, y)
        for character in string:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(character)) # type: ignore