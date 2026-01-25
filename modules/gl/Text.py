import OpenGL.GL as gl
import OpenGL.GLUT as glut

_glut_inited = False

def text_init() -> None:
    return
    global _glut_inited
    if not _glut_inited:
        glut.glutInit()
        _glut_inited = True

def draw_string(x: float, y: float, string: str, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0), big: bool = False)-> None:
    return
    font=glut.GLUT_BITMAP_HELVETICA_12 # type: ignore
    if big:
        font = glut.GLUT_BITMAP_HELVETICA_18 # type: ignore
    gl.glColor4f(*color)
    gl.glRasterPos2f(x, y)
    for character in string:
        glut.glutBitmapCharacter(font, ord(character))
    gl.glRasterPos2f(0, 0)
    gl.glColor4f(1.0, 1.0, 1.0, 1.0)

def draw_box_string(x: float, y: float, string: str, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0), box_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6), big: bool = False)-> None: # type: ignore
    return
    font=glut.GLUT_BITMAP_HELVETICA_12 # type: ignore
    height = 12
    expand = 2
    if big:
        font = glut.GLUT_BITMAP_HELVETICA_18 # type: ignore
        height = 18  # GLUT_BITMAP_HELVETICA_18 is approx 18 pixels high
        expand = 3
    width: int = sum([glut.glutBitmapWidth(font, ord(c)) for c in string])
    # Draw black rectangle behind text
    gl.glColor4f(*box_color)  # semi-transparent black
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(x - expand, y - height - expand)
    gl.glVertex2f(x + width + expand, y - height - expand)
    gl.glVertex2f(x + width + expand, y + expand * 2)
    gl.glVertex2f(x - expand, y + expand * 2)
    gl.glEnd()
    gl.glColor4f(*color)
    gl.glRasterPos2f(x, y)
    for character in string:
        glut.glutBitmapCharacter(font, ord(character))
    gl.glRasterPos2f(0, 0)
    gl.glColor4f(1.0, 1.0, 1.0, 1.0)

