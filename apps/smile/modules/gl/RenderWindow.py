from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT
from threading import Thread
from modules.gl.Utils import FpsCounter
from typing import Callable

class RenderWindow(Thread):
    def __init__(self, width, height, name: str, fullscreen: bool = False, v_sync: bool = True, posX = 0, posY = 0) -> None:
        super().__init__()
        self.window_width: int = width
        self.window_height: int = height
        self.prev_window_width: int = width
        self.prev_window_height: int = height
        self.running = True
        self.fullscreen: bool = fullscreen
        self.v_sync: bool = v_sync
        self.windowName: str = name
        self.fps = FpsCounter()
        self.posX: int = posX
        self.posY: int = posY

        self._is_allocated: bool = False
        self.exit_callback: Callable | None = None

    @property
    def is_allocated(self)-> bool:
        return self._is_allocated


    def run(self) -> None:
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH) # type: ignore
        glut.glutInitWindowSize(self.window_width, self.window_height)
        glut.glutInitWindowPosition(self.posX, self.posY)
        glut.glutCreateWindow(self.windowName.encode())
        if self.fullscreen :
            glut.glutFullScreen()
            glut.glutSetCursor(glut.GLUT_CURSOR_NONE)
        if self.v_sync:
            wglSwapIntervalEXT(1)
        self.init_gl()
        glut.glutDisplayFunc(self.draw_window)
        glut.glutIdleFunc(self.draw_window)
        glut.glutKeyboardFunc(self.keyboard_callback)
        glut.glutSpecialFunc(self.special_key_callback)
        glut.glutReshapeFunc(self.reshape)
        glut.glutCloseFunc(self.close_window)

        version = glGetString(GL_VERSION)
        opengl_version: str = version.decode("utf-8") # type: ignore
        print("OpenGL version:", opengl_version)

        glut.glutMainLoop()

    def stop(self) -> None:
        if self._is_allocated:
            glut.glutLeaveMainLoop()

    def init_gl(self) -> None:
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.setView(self.window_width, self.window_height)

    def reshape(self, width, height) -> None:
        self.window_width = width
        self.window_height = height
        self.setView(self.window_width, self.window_height)

    def setView(self, width, height, flipH = False, flipV = False) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(width if flipH else 0, 0 if flipH else width, 0 if flipV else height, height if flipV else 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)

    def set_fullscreen(self, value: bool) -> None:
        if not self.is_allocated: return
        if self.fullscreen is value: return
        self.fullscreen = value

        if self.fullscreen:
            self.prev_window_width = self.window_width
            self.prev_window_height = self.window_height
            glut.glutFullScreen()
            glut.glutSetCursor(glut.GLUT_CURSOR_NONE)
        else:
            glut.glutReshapeWindow(self.prev_window_width, self.prev_window_height)
            glut.glutPositionWindow(100, 100)
            glut.glutSetCursor(glut.GLUT_CURSOR_LEFT_ARROW)

    def draw_window(self) -> None:
        self.fps.tick()
        fps = str(self.fps.get_fps())
        glut.glutSetWindowTitle((' FPS: ' + fps + '          ' + self.windowName).encode())

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore
        glLoadIdentity()

        self.draw()

        glut.glutSwapBuffers()

        self._is_allocated = True

    def close_window(self,) -> None:
        self._is_allocated = False
        if self.exit_callback: self.exit_callback()

    def draw(self) -> None:
        pass

    def keyboard_callback(self, key, x, y) -> None:
        if key == b'\x1b': # escape
            if self._is_allocated: # redundant?
                glut.glutLeaveMainLoop()

    def special_key_callback(self, key, x, y) -> None:
        pass


    def draw_string(self, x: int, y: int, string: str, flipV=False):


        glRasterPos2f(x, y)
        for character in string:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(character)) # type: ignore