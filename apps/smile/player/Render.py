from OpenGL.GL import * # type: ignore
from threading import Lock
# from cv2 import Mat, imread, resize
import cv2
import numpy as np

from modules.gl.RenderWindow import RenderWindow
from modules.gl.Texture import Texture
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.shaders.Blend import Blend
from modules.gl.Utils import lfo, fit, fill


class Rect():
     def __init__(self, x: int, y: int, width: int, height: int) -> None:
         self.x: int = x
         self.y: int = y
         self.width: int = width
         self.height: int = height

def draw_quad_cw(rect: Rect) -> None :
    x0: float = rect.x
    x1: float = rect.x + rect.width
    y0: float = rect.y
    y1: float = rect.y + rect.height
    glBegin(GL_QUADS)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(x0, y0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(x1, y0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(x1, y1)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(x0, y1)
    glEnd()

def draw_quad_ccw(rect: Rect) -> None :
    x0: float = rect.x
    x1: float = rect.x + rect.width
    y0: float = rect.y
    y1: float = rect.y + rect.height
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(x0, y0)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(x1, y0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(x1, y1)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(x0, y1)
    glEnd()

class Render(RenderWindow):
    def __init__(self, composition_width: int, composition_height: int, img_width: int, img_height: int, ids: list[str], fullscreen: bool = False, v_sync = False, posX: int = 0, posY: int = 0, trail = 0.9) -> None:
        super().__init__(composition_width, composition_height, 'player', fullscreen, v_sync, posX, posY)

        self.input_mutex: Lock = Lock()

        self.width: int = composition_width
        self.height: int = composition_height

        self.img_width: int = img_width
        self.img_height: int = img_height

        self._ids: list[str] = ids
        self._images: dict[str, np.ndarray | None] = {}
        self._initials: dict[str, int] = {}
        self.textures: dict[str, Texture] = {}

        self._rects: dict[str, Rect] = {}

        cntr: int = 0
        for id in ids:
            self._images[id] = None
            self.textures[id] = Texture()
            self._initials[id] = 0

            w = 640
            h = 512
            column: int = cntr % 6
            row: int = cntr // 6
            rect: Rect = Rect(column * w, row * h, w, h)
            self._rects[id] = rect
            cntr += 1

        self.fbos: list[Fbo | SwapFbo] = []
        self.image_fbo = Fbo()
        self.fbos.append(self.image_fbo)
        self.diffusion_fbo = SwapFbo()
        self.fbos.append(self.diffusion_fbo)

        self.shaders: list = []
        self.blend_shader = Blend()
        self.shaders.append(self.blend_shader)

        self.stretch: bool = False

        self.trail: float = trail

        self.white_set: bool = False

    def allocate(self) -> None:
        for fbo in self.fbos: fbo.allocate(self.width, self.height, GL_RGBA32F)
        for shader in self.shaders: shader.allocate(True)

    def reload_shaders(self) -> None :
        for s in self.shaders: s.load(True)

    def draw(self) -> None: # override
        if not self.is_allocated: self.allocate()

        self.update_textures()

        # draw images in FBO
        for id in self._ids:
            texture: Texture = self.textures[id]
            rect: Rect = self._rects[id]
            self.draw_in_fbo_rotated(texture, self.image_fbo, rect)

        self.setView(self.width, self.height, False, True)
        self.update_trails(self.image_fbo, self.diffusion_fbo, self.trail)
        for id in self._ids:
            if self._initials[id] == 2:
                self._initials[id] = 0
                texture: Texture = self.textures[id]
                rect: Rect = self._rects[id]
                self.draw_in_fbo_rotated(texture, self.diffusion_fbo, rect)
            elif self._initials[id] == 1:
                self._initials[id] = 2
                rect: Rect = self._rects[id]
                self.draw_in_fbo_rotated(self.white_texture, self.diffusion_fbo, rect)

        drawFbo: SwapFbo | Fbo = self.diffusion_fbo

        self.setView(self.window_width, self.window_height)
        if self.stretch:
            drawFbo.draw(0, 0, self.window_width, self.window_height)
        else:
            x, y, w, h = fit(self.width, self.height, self.window_width, self.window_height)
            drawFbo.draw(x, y, w, h)

    def update_trails(self, tex: Texture | Fbo | SwapFbo, fbo: SwapFbo, trail: float) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.swap()
        self.blend_shader.use(fbo.fbo_id,
                              tex.tex_id,
                              fbo.back_tex_id,
                              trail)

    def draw_in_fbo(self, tex: Texture | Fbo | SwapFbo, fbo: Fbo | SwapFbo, rect: Rect) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.begin()
        tex.draw(rect.x, rect.y, rect.width, rect.height)
        fbo.end()

    def draw_in_fbo_rotated(self, tex: Texture | Fbo | SwapFbo, fbo: Fbo | SwapFbo, rect: Rect, clockwise: bool = True) -> None:
        self.setView(fbo.width, fbo.height, False, True)
        fbo.begin()
        tex.bind()
        if clockwise:
            draw_quad_cw(rect)
        else:
            draw_quad_ccw(rect)
        tex.unbind()
        fbo.end()


    def update_textures(self) -> None:
        for id in self._ids:
            image: np.ndarray | None = self.get_image(id)
            texture: Texture = self.textures[id]
            if image is not None:
                height, width = image.shape[:2]
                texture.set_from_image(image)
                # if width == self.img_width and  height == self.img_height:
                #     pass
                # #     texture.set_from_image(image)
                # else:
                #     print (width, height, self.img_width, self.img_height)

        if not self.white_set:
            self.white_set = True
            white: np.ndarray = cv2.imread('images/white.jpg')
            self.white_texture: Texture = Texture()
            self.white_texture.set_from_image(white)

    def set_image(self, id: str, image: np.ndarray, initial: bool) -> None :
        with self.input_mutex:
            self._images[id] = image
            self._initials[id] = int(initial)


    def get_image(self, id: str) -> np.ndarray | None :
        with self.input_mutex:
            ret_image: np.ndarray | None = self._images[id]
            self._images[id] = None
            return ret_image

    def keyboard_callback(self, key, x, y) -> None: # override
        super().keyboard_callback(key, x, y)
        if key == b'f' or key == b'F':
            self.set_fullscreen(not self.fullscreen)

        if key == b'r' or key == b'R':
            self.reload_shaders()
