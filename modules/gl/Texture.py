from OpenGL.GL import * # type: ignore


def get_format(internal_format) -> Constant:
    """Get the default pixel format for glTexImage2D (always RGB order).

    Args:
        internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)

    Returns:
        OpenGL format constant (GL_RGB, GL_RGBA, GL_RED, GL_RG)
    """
    if internal_format == GL_RGB: return GL_RGB
    if internal_format == GL_RGB8: return GL_RGB
    if internal_format == GL_RGB16F: return GL_RGB
    if internal_format == GL_RGB32F: return GL_RGB
    if internal_format == GL_RGBA: return GL_RGBA
    if internal_format == GL_RGBA8: return GL_RGBA
    if internal_format == GL_RGBA16F: return GL_RGBA
    if internal_format == GL_RGBA32F: return GL_RGBA
    if internal_format == GL_R8: return GL_RED
    if internal_format == GL_R16F: return GL_RED
    if internal_format == GL_R32F: return GL_RED
    if internal_format == GL_RG32F: return GL_RG
    print('GL_texture', 'internal format not supported')
    return GL_RGB  # fallback


def get_data_type(internal_format) -> Constant:
    if internal_format == GL_RGB: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGB8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGB16F: return GL_FLOAT
    if internal_format == GL_RGB32F: return GL_FLOAT
    if internal_format == GL_RGBA: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGBA8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGBA16F: return GL_FLOAT
    if internal_format == GL_RGBA32F: return GL_FLOAT
    if internal_format == GL_R8: return GL_UNSIGNED_BYTE
    if internal_format == GL_R16F: return GL_FLOAT
    if internal_format == GL_R32F: return GL_FLOAT
    if internal_format == GL_RG32F: return GL_FLOAT
    print('GL_texture', 'internal format not supported')
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

    def allocate(self, width: int, height: int, internal_format,
                 wrap_s: int = GL_CLAMP_TO_EDGE,
                 wrap_t: int = GL_CLAMP_TO_EDGE,
                 min_filter: int = GL_LINEAR,
                 mag_filter: int = GL_LINEAR) -> None :
        """Allocate OpenGL texture with specified dimensions and format.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
            wrap_s: Horizontal wrap mode (default: GL_CLAMP_TO_EDGE)
            wrap_t: Vertical wrap mode (default: GL_CLAMP_TO_EDGE)
            min_filter: Minification filter (default: GL_LINEAR)
            mag_filter: Magnification filter (default: GL_LINEAR)
        """
        data_type: Constant = get_data_type(internal_format)
        if data_type == GL_NONE: return

        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.format = get_format(internal_format)
        self.data_type = data_type
        self.tex_id: int = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)

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

    def bind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, self.tex_id)

    def unbind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, x, y, w, h) -> None :
        self.bind()
        draw_quad(x, y, w, h)
        self.unbind()
