from OpenGL.GL import * # type: ignore
# import cv2
import numpy as np


def get_format(internat_format) -> Constant:
    if internat_format == GL_RGB: return GL_BGR
    if internat_format == GL_RGB8: return GL_BGR
    if internat_format == GL_RGB16F: return GL_BGR
    if internat_format == GL_RGB32F: return GL_BGR
    if internat_format == GL_RGBA: return GL_BGRA
    if internat_format == GL_RGBA8: return GL_BGRA
    if internat_format == GL_RGBA16F: return GL_BGRA
    if internat_format == GL_RGBA32F: return GL_BGRA
    if internat_format == GL_R8: return GL_RED
    if internat_format == GL_R16F: return GL_RED
    if internat_format == GL_R32F: return GL_RED
    print('GL_texture', 'internal format not supported')
    return GL_NONE

def get_data_type(internat_format) -> Constant:
    if internat_format == GL_RGB: return GL_UNSIGNED_BYTE
    if internat_format == GL_RGB8: return GL_UNSIGNED_BYTE
    if internat_format == GL_RGB16F: return GL_FLOAT
    if internat_format == GL_RGB32F: return GL_FLOAT
    if internat_format == GL_RGBA: return GL_UNSIGNED_BYTE
    if internat_format == GL_RGBA8: return GL_UNSIGNED_BYTE
    if internat_format == GL_RGBA16F: return GL_FLOAT
    if internat_format == GL_RGBA32F: return GL_FLOAT
    if internat_format == GL_R8: return GL_UNSIGNED_BYTE
    if internat_format == GL_R16F: return GL_FLOAT
    if internat_format == GL_R32F: return GL_FLOAT
    print('GL_texture', 'internal format not supported')
    return GL_NONE

def get_internal_format(image: np.ndarray) -> Constant:
    # only works for byte images (not float)
    if image.dtype == np.uint8:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R8
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB8
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA8
    elif image.dtype == np.float16:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R16F
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB16F
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA16F
    elif image.dtype == np.float32:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R32F
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB32F
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA32F

    print('GL_texture', 'image format not supported')
    return GL_NONE

def draw_quad(x: float, y: float, w: float, h: float, flipV: bool = False) -> None :
    x0: float = x
    x1: float = x + w
    y0: float = y
    y1: float = y + h

    glBegin(GL_QUADS)

    if flipV:
        # Lower-left corner
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x0, y0)

        # Lower-right corner
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x1, y0)

        # Upper-right corner
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x1, y1)

        # Upper-left corner
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x0, y1)

    else:
        # Lower-left corner
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x0, y0)

        # Lower-right corner
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x1, y0)

        # Upper-right corner
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x1, y1)

        # Upper-left corner
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x0, y1)

    glEnd()

class Texture():
    def __init__(self) -> None :
        self.allocated = False
        self.width: int = 0
        self.height: int = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        self.tex_id = 0

    def allocate(self, width: int, height: int, internal_format) -> None :
        data_type: Constant = get_data_type(internal_format)
        if data_type == GL_NONE: return

        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.format = get_format(internal_format)
        self.data_type = data_type
        self.tex_id: int = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        # Set the swizzle mask for the texture to draw it as grayscale
        if self.format == GL_RED:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, self.width, self.height, 0, self.format, self.data_type, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.allocated = True

    def deallocate(self) -> None :
        if not self.allocated: return
        self.allocated = False
        self.width = 0
        self.height = 0
        self.internal_format = GL_NONE
        self.format = GL_NONE
        self.data_type = GL_NONE
        glDeleteTextures(1, [self.tex_id])
        self.tex_id = 0

    def set_from_image(self, image: np.ndarray) -> None:
        internal_format: Constant = get_internal_format(image)
        if internal_format == GL_NONE: return
        height: int = image.shape[0]
        width:  int = image.shape[1]

        if internal_format != self.internal_format or width != self.width or height != self.height:
            if self.allocated: self.deallocate()
            self.allocate(width, height, internal_format)

        if not self.allocated: return

        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, width, height, 0, self.format, self.data_type, image)
        self.unbind()

    def bind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, self.tex_id)

    def unbind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, x, y, w, h) -> None :
        self.bind()
        draw_quad(x, y, w, h)
        self.unbind()
