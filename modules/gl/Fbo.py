from OpenGL.GL import * # type: ignore
from modules.gl.Texture import Texture

class Fbo(Texture):
    def __init__(self) -> None :
        super(Fbo, self).__init__()
        self.fbo_id = 0

    def allocate(self, width: int, height: int, internal_format) -> None :
        super(Fbo, self).allocate(width, height, internal_format)
        if not self.allocated: return

        self.fbo_id = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        self.bind()
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex_id, 0)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.unbind()

    def begin(self)  -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        # Apply transformation to flip the y-coordinates
        # glPushMatrix()
        # glTranslatef(0, self.height, 0)
        # glScalef(1, -1, 1)

    def end(self)  -> None:
        # glPopMatrix()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class SwapFbo():
    def __init__(self) -> None :
        self.width: int = 0
        self.height: int = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        self.fbos: list[Fbo] = [Fbo(), Fbo()]
        self.swap_state: bool = False
        self.allocated: bool = False
        self.fbo_id = 0
        self.tex_id = 0
        self.back_tex_id = 0

    def allocate(self, width: int, height: int, internal_format) -> None :
        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.fbos[0].allocate(width, height, internal_format)
        self.fbos[1].allocate(width, height, internal_format)
        self.format = self.fbos[0].format
        self.data_type = self.fbos[0].data_type
        self.allocated = self.fbos[0].allocated and self.fbos[1].allocated

        self.fbo_id = self.fbos[self.swap_state].fbo_id
        self.tex_id = self.fbos[self.swap_state].tex_id
        self.back_tex_id = self.fbos[not self.swap_state].tex_id

    def swap(self) -> None :
        self.swap_state = not self.swap_state
        self.fbo_id = self.fbos[self.swap_state].fbo_id
        self.tex_id = self.fbos[self.swap_state].tex_id
        self.back_tex_id = self.fbos[not self.swap_state].tex_id

    def begin(self) -> None :
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

    def end(self) -> None :
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, self.tex_id)

    def unbind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, x, y, w, h) -> None :
        self.fbos[self.swap_state].draw(x, y, w, h)

    def draw_back(self, x, y, w, h) -> None :
        self.fbos[not self.swap_state].draw(x, y, w, h)